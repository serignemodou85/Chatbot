# ── src/agents/crew.py ───────────────────────────────────────────────────────
# Orchestrateur Phase 2 : route la question, fait le RAG en Python,
# passe le contexte au ReportAgent pour synthèse.
#
# Architecture :
#   1. _classify()  — routage par mots-clés (Python, ~0ms)
#   2. RAG search   — retrieval ciblé par outil (Python, ~1s)
#   3. ReportAgent  — un seul appel LLM pour la synthèse (~1-2min)

from __future__ import annotations

import re
from typing import Any

from crewai import Crew, Process
from loguru import logger

from src.agents.agents import make_report_agent
from src.agents.tasks import make_report_task
from src.retrieval.rag_chain import _sanitize_input
from src.retrieval.reranker import rerank
from src.retrieval.cache import SemanticCache
from src.retrieval.command_validator import validate_commands, validate_length

_crew_cache: SemanticCache = SemanticCache()

# ── Mots-clés pour le routage dynamique ──────────────────────────────────────
_NETWORK_KW = {
    "vpn", "réseau", "reseau", "network", "pare-feu", "parefeu", "firewall",
    "pfsense", "vlan", "routeur", "router", "tcp", "udp", "dns", "dhcp",
    "openvpn", "ssh", "port", "interface", "nat", "wan", "lan", "ip",
    "connectivité", "connectivity", "bande passante", "bandwidth",
    "ipsec", "ikev1", "ikev2", "ike", "tunnel", "bgp", "ospf", "gre",
    "l2tp", "wireguard", "switch", "commutateur", "routage", "acl", "qos",
}

_SECURITY_KW = {
    "wazuh", "siem", "intrusion", "alerte", "alert", "cve", "zabbix",
    "vulnérabilité", "vulnerability", "audit", "soc", "supervision",
    "monitoring", "log", "incident", "malware", "détection", "detection",
    "règle", "rule", "agent", "sécurité", "securite", "cybersecu",
    "attaque", "attack", "menace", "threat", "compliance", "conformité",
}

# Outils détectables dans la question pour recherches ciblées
_TOOL_SEARCHES = [
    # (keyword dans question, query_prefix, source_keywords, label)
    ("pfsense",    "pfsense firewall configuration",  ["pfsense", "openvpn"],        "pfSense"),
    ("openvpn",    "openvpn vpn tunnel",              ["pfsense", "vpn", "openvpn"], "OpenVPN"),
    ("ipsec",      "ipsec IKE_SA phase1 mismatch PSK pre-shared key NO_PROPOSAL_CHOSEN NAT-T troubleshoot", ["pfsense", "vpn", "ipsec"],   "IPSec"),
    ("ike",        "ipsec IKE phase1 hash encryption algorithm mismatch PSK pre-shared key NAT-T UDP 500", ["pfsense", "vpn", "ipsec"],   "IPSec"),
    ("tunnel",     "ipsec vpn tunnel IKE_SA CHILD_SA phase1 phase2 troubleshoot logs",                     ["pfsense", "vpn", "openvpn"], "VPN"),
    ("wazuh",      "wazuh siem configuration",        ["wazuh"],                      "Wazuh"),
    ("zabbix",     "zabbix monitoring alert",         ["zabbix"],                     "Zabbix"),
    ("siem",       "siem detection rules",            ["wazuh", "siem"],              "SIEM"),
    ("ubuntu",     "ubuntu server administration",    ["ubuntu"],                     "Ubuntu"),
    ("linux",      "linux administration command",    ["linux", "ubuntu"],            "Linux"),
    ("windows",    "windows server administration",   ["windows"],                    "Windows"),
    ("dolibarr",   "dolibarr utilisateur permission module configuration sécurité", ["dolibarr"], "Dolibarr"),
    ("ssh",        "SSH access management firewall rule source IP restrict",       ["pfsense", "linux", "ubuntu"], "SSH"),
    ("firewall",   "pfsense firewall rule pass block interface source destination alias", ["pfsense"], "Firewall"),
    ("spoofing",   "anti-spoofing bogon block private reserved IP pfSense interface",     ["pfsense"], "Anti-spoofing"),
    ("règle",      "firewall rule configuration source destination interface pass block",  ["pfsense"], "Règle FW"),
]


def _classify(question: str) -> list[str]:
    """Route la question vers un ou plusieurs domaines."""
    q = question.lower()
    words = set(q.split())
    domains: list[str] = []

    if words & _NETWORK_KW or any(kw in q for kw in _NETWORK_KW):
        domains.append("network")
    if words & _SECURITY_KW or any(kw in q for kw in _SECURITY_KW):
        domains.append("security")
    if not domains:
        domains.append("doc")

    return domains


def _targeted_search(vs, question: str, query_prefix: str, source_keywords: list[str], k: int = 4) -> list:
    """Recherche ciblée pour un outil spécifique.

    Deux passes de similarity_search (queries complémentaires) pour maximiser le recall.
    Le cross-encoder est bypassed ici : il est anglophone et fait remonter les changelogs
    devant les sections de troubleshooting quand la question est en français.
    On utilise donc l'ordre cosine direct (déjà pertinent car les requêtes sont en anglais).
    """
    q_combined = f"{query_prefix} {question}"
    r1 = vs.similarity_search(q_combined, k=20)
    r2 = vs.similarity_search(query_prefix, k=15)

    # Troisième passe : messages d'erreur bruts (charon logs) — ignore la question FR
    # Cible les chunks de troubleshooting qui commencent en milieu de contexte
    r3_query = "charon IKE phase 1 NO_PROPOSAL_CHOSEN authentication failed decryption PSK mismatch hash"
    r3 = vs.similarity_search(r3_query, k=15)

    # r3 en priorité : contient les chunks d'erreurs brutes (NO_PROPOSAL_CHOSEN, etc.)
    # qui sont souvent en milieu de section et sous-représentés par cosine seul.
    seen: set[str] = set()
    merged: list = []
    for d in r3 + r1 + r2:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            merged.append(d)

    filtered = [
        d for d in merged
        if any(kw in d.metadata.get("source", "").lower() for kw in source_keywords)
        and not _is_changelog(d)
        and len(d.page_content.strip()) > 80  # exclut les chunks ToC (1 ligne)
    ]
    candidates = filtered if filtered else [d for d in merged if not _is_changelog(d)]

    # Max 2 chunks par page pour garantir la diversité du contexte
    page_count: dict[int, int] = {}
    diverse: list = []
    for d in candidates:
        p = d.metadata.get("page", -1)
        if page_count.get(p, 0) < 2:
            diverse.append(d)
            page_count[p] = page_count.get(p, 0) + 1
        if len(diverse) >= k:
            break

    return diverse


_CHANGELOG_PATTERNS = re.compile(
    r"(^## \d+\.\d+\.\d|Fixed:|Changed:|Added:|Improved:|•\s*Fix|•\s*Chang|#\d{4,})",
    re.MULTILINE,
)


def _is_changelog(doc) -> bool:
    """Détecte les chunks de release notes / changelogs — inutiles pour des questions opérationnelles."""
    text = doc.page_content
    matches = _CHANGELOG_PATTERNS.findall(text)
    return len(matches) >= 3


def _dedup(docs: list) -> list:
    """Supprime les chunks quasi-identiques (mêmes 80 premiers chars)."""
    seen: set[str] = set()
    unique: list = []
    for d in docs:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


class CyberSecCrew:
    """
    Orchestre la recherche documentaire et la synthèse LLM.

    Le retrieval est fait en Python (fiable, rapide).
    Le LLM intervient uniquement pour la synthèse finale.
    """

    def run(self, question: str, user_mode: str = "🖥️ Admin système") -> dict[str, Any]:
        if len(question) > 4000:
            return {
                "question": question,
                "user_mode": user_mode,
                "domains": [],
                "agents_used": [],
                "report": "Question trop longue (maximum 4000 caractères).",
            }
        question = _sanitize_input(question)

        cached = _crew_cache.get(question, user_mode)
        if cached is not None:
            return cached

        domains = _classify(question)
        logger.info(f"[Crew] Question routée vers : {domains}")

        from src.agents.tools.rag_tools import (
            _format_docs,
            _get_vsm,
            _NETWORK_SOURCE_KEYWORDS,
            _SECURITY_SOURCE_KEYWORDS,
        )

        vsm = _get_vsm()
        vs  = vsm._vectorstore
        q_lower = question.lower()
        context_parts: list[str] = []

        # ── Détection des outils mentionnés dans la question ──────────────────
        tool_docs: dict[str, list] = {}
        for tool_kw, query_prefix, source_kws, label in _TOOL_SEARCHES:
            if tool_kw in q_lower:
                docs = _targeted_search(vs, question, query_prefix, source_kws, k=4)
                if docs:
                    tool_docs[label] = docs
                    logger.info(f"[Crew] Recherche {label} : {len(docs)} chunks")

        # ── Recherche globale (partagée entre network et security) ───────────────
        shared_search: list | None = None
        if "network" in domains or "security" in domains:
            shared_search = vs.similarity_search(question, k=25)

        # ── Assemblage du contexte par domaine ────────────────────────────────
        if "doc" in domains:
            # Si un outil spécifique (Dolibarr...) a été détecté, utiliser ses chunks ciblés
            doc_tool_labels = ["Dolibarr"]
            doc_tool_docs: list = []
            for label in doc_tool_labels:
                doc_tool_docs.extend(tool_docs.get(label, []))

            if doc_tool_docs:
                context_parts.append(
                    f"=== DOCUMENTATION SPÉCIFIQUE ===\n{_format_docs(doc_tool_docs)}"
                )
                logger.info(f"[Crew] Recherche doc ciblée : {len(doc_tool_docs)} chunks")
            else:
                docs = vsm.get_retriever(k=6).invoke(question)
                if docs:
                    context_parts.append(
                        f"=== DOCUMENTATION GÉNÉRALE ===\n{_format_docs(docs)}"
                    )
                logger.info(f"[Crew] Recherche doc générale : {len(docs if not doc_tool_docs else [])} chunks")

        if "network" in domains:
            net_labels = ["pfSense", "OpenVPN", "Ubuntu", "Windows", "Linux", "SSH", "Firewall", "Anti-spoofing", "Règle FW"]
            net_docs: list = []

            # Chunks des outils réseau trouvés
            for label in net_labels:
                net_docs.extend(tool_docs.get(label, []))

            # Complément si pas assez de chunks spécifiques
            if len(net_docs) < 4:
                fallback = [
                    d for d in (shared_search or [])
                    if any(kw in d.metadata.get("source", "").lower()
                           for kw in _NETWORK_SOURCE_KEYWORDS)
                ]
                net_docs.extend(fallback[:6])

            net_docs = _dedup(net_docs)[:12]
            if len(net_docs) > 1:
                net_docs = rerank(question, net_docs, top_k=6)
            if net_docs:
                context_parts.append(
                    f"=== RÉSEAU & INFRASTRUCTURE ===\n{_format_docs(net_docs)}"
                )
            logger.info(f"[Crew] Contexte réseau : {len(net_docs)} chunks (re-rankés)")

        if "security" in domains:
            sec_labels = ["Wazuh", "Zabbix", "SIEM"]
            sec_docs: list = []

            # Chunks des outils sécurité trouvés
            for label in sec_labels:
                sec_docs.extend(tool_docs.get(label, []))

            # Complément si pas assez de chunks
            if len(sec_docs) < 4:
                fallback = [
                    d for d in (shared_search or [])
                    if any(kw in d.metadata.get("source", "").lower()
                           for kw in _SECURITY_SOURCE_KEYWORDS)
                ]
                sec_docs.extend(fallback[:6])

            sec_docs = _dedup(sec_docs)[:12]
            if len(sec_docs) > 1:
                sec_docs = rerank(question, sec_docs, top_k=6)
            if sec_docs:
                context_parts.append(
                    f"=== CYBERSÉCURITÉ & SIEM ===\n{_format_docs(sec_docs)}"
                )
            logger.info(f"[Crew] Contexte sécurité : {len(sec_docs)} chunks (re-rankés)")

        combined_context = (
            "\n\n".join(context_parts)
            if context_parts
            else "Aucune information trouvée dans la documentation pour cette question."
        )

        # ── Rapport final — un seul appel LLM ─────────────────────────────────
        logger.info("[Crew] Génération du rapport final...")
        report_agent = make_report_agent()
        report_task  = make_report_task(
            agent=report_agent,
            question=question,
            user_mode=user_mode,
            context=combined_context,
        )
        report_crew = Crew(
            agents=[report_agent],
            tasks=[report_task],
            process=Process.sequential,
            verbose=False,
        )
        final_output = report_crew.kickoff()
        validated_report = validate_commands(str(final_output), combined_context, question)
        validated_report = validate_length(validated_report)

        result = {
            "question": question,
            "user_mode": user_mode,
            "domains": domains,
            "agents_used": domains,
            "report": validated_report,
        }
        _crew_cache.set(question, user_mode, result)
        return result
