# ── src/agents/crew.py ───────────────────────────────────────────────────────
# Orchestrateur Phase 2 : route la question vers les agents compétents,
# lance le crew, retourne le rapport final.
#
# Utilisation :
#   from src.agents.crew import CyberSecCrew
#   result = CyberSecCrew().run("Compare Wazuh et Zabbix", "🖥️ Admin système")

from __future__ import annotations

from typing import Any

from crewai import Crew, Process
from loguru import logger

from src.agents.agents import (
    make_doc_agent,
    make_network_agent,
    make_report_agent,
    make_security_agent,
)
from src.agents.tasks import make_report_task, make_research_task

# ── Mots-clés pour le routage dynamique ──────────────────────────────────────
_NETWORK_KW = {
    "vpn", "réseau", "reseau", "network", "pare-feu", "parefeu", "firewall",
    "pfsense", "vlan", "routeur", "router", "tcp", "udp", "dns", "dhcp",
    "openvpn", "ssh", "port", "interface", "nat", "wan", "lan", "ip",
    "connectivité", "connectivity", "bande passante", "bandwidth",
}

_SECURITY_KW = {
    "wazuh", "siem", "intrusion", "alerte", "alert", "cve", "zabbix",
    "vulnérabilité", "vulnerability", "audit", "soc", "supervision",
    "monitoring", "log", "incident", "malware", "détection", "detection",
    "règle", "rule", "agent", "sécurité", "securite", "cybersecu",
    "attaque", "attack", "menace", "threat", "compliance", "conformité",
}


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


class CyberSecCrew:
    """
    Orchestre les agents spécialisés pour répondre à une question complexe.

    Étape 1 — Classification : détermine quels agents invoquer (Python, rapide)
    Étape 2 — Recherche      : les agents sélectionnés interrogent ChromaDB
    Étape 3 — Rapport        : ReportAgent agrège et formate le résultat final
    """

    def run(self, question: str, user_mode: str = "🖥️ Admin système") -> dict[str, Any]:
        domains = _classify(question)
        logger.info(f"[Crew] Question routée vers : {domains}")

        # ── Construction dynamique des agents et tâches de recherche ─────────
        research_agents = []
        research_tasks = []

        if "doc" in domains:
            a = make_doc_agent()
            research_agents.append(a)
            research_tasks.append(
                make_research_task(a, question, "Documentation générale IT et système")
            )
        if "network" in domains:
            a = make_network_agent()
            research_agents.append(a)
            research_tasks.append(
                make_research_task(a, question, "Réseau, VPN, pare-feu, infrastructure")
            )
        if "security" in domains:
            a = make_security_agent()
            research_agents.append(a)
            research_tasks.append(
                make_research_task(a, question, "Cybersécurité, SIEM, Wazuh, Zabbix")
            )

        # ── Phase recherche ───────────────────────────────────────────────────
        logger.info(f"[Crew] Lancement de {len(research_agents)} agent(s) de recherche...")
        research_crew = Crew(
            agents=research_agents,
            tasks=research_tasks,
            process=Process.sequential,
            verbose=True,
        )
        research_output = research_crew.kickoff()

        # ── Phase rapport ─────────────────────────────────────────────────────
        logger.info("[Crew] Génération du rapport final...")
        report_agent = make_report_agent()
        report_task = make_report_task(
            agent=report_agent,
            question=question,
            user_mode=user_mode,
            context=str(research_output),
        )
        report_crew = Crew(
            agents=[report_agent],
            tasks=[report_task],
            process=Process.sequential,
            verbose=False,
        )
        final_output = report_crew.kickoff()

        return {
            "question": question,
            "user_mode": user_mode,
            "domains": domains,
            "agents_used": [a.role for a in research_agents],
            "report": str(final_output),
        }
