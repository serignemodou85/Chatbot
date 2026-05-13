# ── src/ingestion/vectorstore.py ─────────────────────────────────────────────
# Responsabilité : créer et gérer la base vectorielle ChromaDB.
#
# Deux opérations principales :
#   1. build()  → indexation initiale (à lancer une seule fois)
#   2. load()   → chargement de la base existante (à chaque démarrage)
#
# Utilisation :
#   vs = VectorStoreManager()
#   vs.build(chunks)          # Première fois
#   retriever = vs.load().as_retriever()  # Ensuite

from typing import List, Optional

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from config.settings import settings
from src.ingestion.deduplication import deduplicate_chunks


def get_embedding_model():
    """
    Retourne le modèle d'embedding configuré dans .env.

    HuggingFace (défaut) : gratuit, tourne en local, pas d'API key.
    OpenAI : meilleure qualité sémantique, mais payant à l'usage.
    """
    if settings.EMBEDDING_PROVIDER == "openai":
        logger.info("Embeddings : OpenAI text-embedding-ada-002")
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=settings.OPENAI_API_KEY,
        )

    # Défaut : HuggingFace, entièrement local
    logger.info(f"Embeddings : HuggingFace {settings.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        # Cache le modèle localement pour éviter de le retélécharger
        cache_folder="./data/embedding_cache",
    )


class VectorStoreManager:
    """
    Gère le cycle de vie de la base vectorielle ChromaDB.

    ChromaDB persiste les données sur disque dans CHROMA_PERSIST_DIR,
    ce qui permet de ne faire l'indexation qu'une seule fois.
    """

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.CHROMA_COLLECTION
        self._vectorstore: Optional[Chroma] = None

    def build(self, chunks: List[Document]) -> Chroma:
        """
        Indexe les chunks dans ChromaDB.
        À appeler une seule fois lors de la création initiale
        ou lors d'une mise à jour complète de la base.

        ⚠️  Cette opération écrase la collection existante.
        """
        if not chunks:
            raise ValueError("Aucun chunk à indexer. Vérifier le dossier data/docs/")

        # Déduplication systématique avant indexation pour éviter les doublons
        # en cas de documents réindexés plusieurs fois.
        before_dedup = len(chunks)
        chunks = deduplicate_chunks(chunks, similarity_threshold=0.92)
        removed = before_dedup - len(chunks)
        if removed > 0:
            logger.info(f"Déduplication build(): {removed} chunk(s) supprimé(s)")

        logger.info(f"Indexation de {len(chunks)} chunks dans ChromaDB...")

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )

        count = self._vectorstore._collection.count()
        logger.info(f"Base vectorielle créée : {count} vecteurs stockés dans {self.persist_dir}")
        return self._vectorstore

    def load(self) -> Chroma:
        """
        Charge une base ChromaDB existante depuis le disque.
        Lève une erreur si la base n'a pas encore été créée via build().
        """
        import os
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"Base ChromaDB introuvable : {self.persist_dir}\n"
                "Lancer d'abord : python scripts/build_index.py"
            )

        logger.info(f"Chargement ChromaDB depuis {self.persist_dir}...")
        self._vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )

        count = self._vectorstore._collection.count()
        logger.info(f"Base chargée : {count} vecteurs disponibles")
        return self._vectorstore

    def get_retriever(self, k: int = None):
        """
        Retourne un retriever LangChain prêt à l'emploi.
        k = nombre de chunks retournés par requête.
        """
        if self._vectorstore is None:
            self.load()

        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or settings.RETRIEVER_K},
        )

    def add_documents(self, chunks: List[Document]):
        """
        Ajoute de nouveaux documents à une base existante
        sans écraser ce qui est déjà indexé.
        """
        if self._vectorstore is None:
            self.load()

        self._vectorstore.add_documents(chunks)
        logger.info(f"{len(chunks)} nouveaux chunks ajoutés à la base.")
