# ── src/agents/agents.py ─────────────────────────────────────────────────────
# Définit l'agent de synthèse Phase 2.
# Le retrieval est fait directement en Python (crew.py) — le ReportAgent
# synthétise uniquement, sans outil de recherche.

from crewai import Agent
from langchain_community.chat_models import ChatOllama

from config.settings import settings


def _crew_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.CREW_LLM_MODEL,
        temperature=0.1,
        base_url=settings.OLLAMA_BASE_URL,
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
            "ton niveau de détail et ton vocabulaire au profil de l'utilisateur. "
            "RÈGLE ABSOLUE : tu n'écris jamais une commande, un nom de fichier ou une "
            "procédure qui n'apparaît pas textuellement dans le contexte documentaire "
            "fourni dans la tâche. Si une information n'est pas dans le contexte, "
            "tu le signales explicitement au lieu d'inventer."
        ),
        tools=[],
        llm=_crew_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=5,
    )
