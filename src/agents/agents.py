# ── src/agents/agents.py ─────────────────────────────────────────────────────
# Définit les 4 agents spécialisés de la Phase 2.
# Chaque agent a un rôle, un objectif et un outil de recherche dédié.
# Le ReportAgent n'a pas d'outil — il synthétise uniquement.

from crewai import Agent
from langchain_community.chat_models import ChatOllama

from config.settings import settings
from src.agents.tools.rag_tools import (
    DocumentationSearchTool,
    NetworkSearchTool,
    SecuritySearchTool,
)


def _crew_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.CREW_LLM_MODEL,
        temperature=0.1,
        base_url=settings.OLLAMA_BASE_URL,
    )


def make_doc_agent() -> Agent:
    return Agent(
        role="Expert en Documentation Technique IT",
        goal=(
            "Trouver dans la documentation les informations précises pour répondre "
            "à la question posée, en extrayant procédures, commandes et concepts clés."
        ),
        backstory=(
            "Tu es un spécialiste de la documentation technique avec 10 ans d'expérience "
            "en administration système Linux et Windows. Tu excelles à localiser "
            "l'information exacte dans des milliers de pages de documentation."
        ),
        tools=[DocumentationSearchTool()],
        llm=_crew_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def make_network_agent() -> Agent:
    return Agent(
        role="Spécialiste Réseau et Infrastructure",
        goal=(
            "Analyser les aspects réseau de la question et fournir des configurations "
            "précises, des commandes opérationnelles et des procédures step-by-step."
        ),
        backstory=(
            "Tu es un ingénieur réseau senior, expert en pfSense, OpenVPN, pare-feux, "
            "routage et protocoles réseau. Tu as configuré des centaines d'infrastructures "
            "réseau et tu fournis toujours des réponses concrètes et vérifiables."
        ),
        tools=[NetworkSearchTool()],
        llm=_crew_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def make_security_agent() -> Agent:
    return Agent(
        role="Expert Cybersécurité et SIEM",
        goal=(
            "Analyser les enjeux de sécurité, identifier les vulnérabilités et "
            "formuler des recommandations défensives basées sur les meilleures pratiques."
        ),
        backstory=(
            "Tu es un analyste SOC niveau 3, expert Wazuh et Zabbix, avec une expérience "
            "en réponse à incidents et en déploiement de SIEM. Tu penses toujours "
            "en termes de risque, de détection et de remédiation."
        ),
        tools=[SecuritySearchTool()],
        llm=_crew_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def make_report_agent() -> Agent:
    return Agent(
        role="Rédacteur de Rapports Techniques",
        goal=(
            "Agréger les analyses des agents spécialistes et produire un rapport "
            "final structuré, clair et directement exploitable par l'utilisateur."
        ),
        backstory=(
            "Tu es expert en communication technique. Tu synthétises des analyses "
            "complexes en rapports Markdown bien structurés. Tu adaptes toujours "
            "ton niveau de détail et ton vocabulaire au profil de l'utilisateur."
        ),
        tools=[],
        llm=_crew_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=5,
    )
