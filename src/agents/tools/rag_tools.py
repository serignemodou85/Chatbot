# ── src/agents/tools/rag_tools.py ────────────────────────────────────────────
# Trois outils CrewAI qui wrappent ChromaDB avec filtrage par domaine.
# Un seul vectorstore partagé (singleton) — pas de réindexation nécessaire.
# Chaque outil filtre les résultats par mots-clés dans le chemin source.

from __future__ import annotations

from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.ingestion.vectorstore import VectorStoreManager

# ── Singleton vectorstore ─────────────────────────────────────────────────────
_vsm: VectorStoreManager | None = None


def _get_vsm() -> VectorStoreManager:
    global _vsm
    if _vsm is None:
        _vsm = VectorStoreManager()
        _vsm.load()
    return _vsm


def _format_docs(docs: list) -> str:
    if not docs:
        return "Aucun document pertinent trouvé dans la base documentaire."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Source inconnue")
        page = doc.metadata.get("page", "")
        page_info = f" — page {page}" if page else ""
        parts.append(
            f"[Doc {i}] {source}{page_info}\n{doc.page_content[:500]}"
        )
    return "\n\n---\n\n".join(parts)


# ── Schéma d'entrée commun ────────────────────────────────────────────────────
class SearchInput(BaseModel):
    query: str = Field(description="La requête de recherche en langage naturel")


# ── Outil Documentation — tous les docs ──────────────────────────────────────
class DocumentationSearchTool(BaseTool):
    name: str = "documentation_search"
    description: str = (
        "Recherche dans toute la documentation technique : administration système, "
        "Linux, Windows Server, Ubuntu, Dolibarr, guides d'installation. "
        "À utiliser pour les questions générales IT et système."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        vsm = _get_vsm()
        retriever = vsm.get_retriever(k=5)
        docs = retriever.invoke(query)
        return _format_docs(docs)


# ── Outil Réseau — filtrage sur docs réseau/VPN/firewall ─────────────────────
_NETWORK_SOURCE_KEYWORDS = ["pfsense", "vpn", "ubuntu", "network", "firewall", "windows"]


class NetworkSearchTool(BaseTool):
    name: str = "network_search"
    description: str = (
        "Recherche dans la documentation réseau et infrastructure : "
        "VPN, pare-feu, pfSense, OpenVPN, routage, VLAN, protocoles réseau, "
        "configuration d'interfaces. À utiliser pour les questions réseau."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        vsm = _get_vsm()
        vs = vsm._vectorstore
        all_docs = vs.similarity_search(query, k=12)
        filtered = [
            d for d in all_docs
            if any(kw in d.metadata.get("source", "").lower() for kw in _NETWORK_SOURCE_KEYWORDS)
        ]
        docs = filtered[:5] if filtered else all_docs[:5]
        return _format_docs(docs)


# ── Outil Sécurité — filtrage sur docs Wazuh/Zabbix/SIEM ────────────────────
_SECURITY_SOURCE_KEYWORDS = ["wazuh", "zabbix", "siem", "security", "securit"]


class SecuritySearchTool(BaseTool):
    name: str = "security_search"
    description: str = (
        "Recherche dans la documentation cybersécurité : "
        "Wazuh, Zabbix, SIEM, détection d'intrusions, supervision sécurité, "
        "gestion des alertes, corrélation d'événements. "
        "À utiliser pour les questions de sécurité et monitoring."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        vsm = _get_vsm()
        vs = vsm._vectorstore
        all_docs = vs.similarity_search(query, k=12)
        filtered = [
            d for d in all_docs
            if any(kw in d.metadata.get("source", "").lower() for kw in _SECURITY_SOURCE_KEYWORDS)
        ]
        docs = filtered[:5] if filtered else all_docs[:5]
        return _format_docs(docs)
