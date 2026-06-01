# ── src/llm/llm_factory.py ───────────────────────────────────────────────────
# Responsabilité : instancier le bon LLM selon la config .env.
# Pattern Factory : le reste du code n'a pas besoin de savoir
# si on utilise OpenAI ou Ollama — il appelle juste get_llm().
#
# Utilisation :
#   from src.llm.llm_factory import get_llm
#   llm = get_llm()

from langchain_core.language_models import BaseChatModel
from loguru import logger

from config.settings import settings


def get_llm() -> BaseChatModel:
    """
    Retourne une instance LLM configurée selon LLM_PROVIDER dans .env.

    OpenAI (LLM_PROVIDER=openai) :
      - Nécessite OPENAI_API_KEY
      - Modèle recommandé : gpt-4o
      - Avantage : meilleure qualité, pas d'installation locale

    Ollama (LLM_PROVIDER=ollama) :
      - Nécessite Ollama installé localement (ollama.ai)
      - Modèle recommandé : llama3
      - Avantage : gratuit, privé, fonctionne hors ligne
      - Installer le modèle : ollama pull llama3
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info(f"LLM : OpenAI {settings.LLM_MODEL}")
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
            # Streaming activé pour afficher la réponse mot par mot
            streaming=True,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        logger.info(f"LLM : Ollama {settings.LLM_MODEL} ({settings.OLLAMA_BASE_URL})")
        return ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            base_url=settings.OLLAMA_BASE_URL,
        )

    raise ValueError(
        f"LLM_PROVIDER inconnu : '{provider}'. "
        "Valeurs acceptées : 'openai' ou 'ollama'"
    )
