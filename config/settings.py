# ── config/settings.py ───────────────────────────────────────────────────────
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:
    """Paramètres globaux du projet, lus depuis .env."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str      = os.getenv("LLM_PROVIDER", "ollama")
    LLM_MODEL: str         = os.getenv("LLM_MODEL", "llama3")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "")
    OLLAMA_BASE_URL: str   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Phase 2 — CrewAI ─────────────────────────────────────────────────────
    CREW_LLM_MODEL: str = os.getenv("CREW_LLM_MODEL", "llama3")

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL: str    = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION: str  = os.getenv("CHROMA_COLLECTION", "cybersec_docs")

    # ── RAG ───────────────────────────────────────────────────────────────────
    CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    RETRIEVER_K: int   = int(os.getenv("RETRIEVER_K", "4"))

    # ── Dossiers ──────────────────────────────────────────────────────────────
    DOCS_DIR: str = os.getenv("DOCS_DIR", "./data/docs")

    def validate(self):
        """Vérifie que la config est cohérente au démarrage."""
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=openai mais OPENAI_API_KEY est vide. "
                "Vérifier votre fichier .env"
            )
        if self.LLM_PROVIDER not in ("openai", "ollama"):
            raise ValueError(f"LLM_PROVIDER inconnu : {self.LLM_PROVIDER}")


settings = Settings()