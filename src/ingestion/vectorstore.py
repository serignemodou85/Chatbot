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

    # ChromaDB refuse les batches > 5461 items (limite interne SQLite)
    CHROMA_BATCH_SIZE = 5000

    def build(self, chunks: List[Document]) -> Chroma:
        """
        Indexe les chunks dans ChromaDB par batches de 5000.

        ChromaDB 0.5.x limite les upserts à 5461 items par appel.
        On crée la collection avec le premier batch, puis on ajoute les suivants.
        """
        if not chunks:
            raise ValueError("Aucun chunk à indexer. Vérifier le dossier data/docs/")

        before_dedup = len(chunks)
        chunks = deduplicate_chunks(chunks, similarity_threshold=0.92)
        removed = before_dedup - len(chunks)
        if removed > 0:
            logger.info(f"Déduplication build(): {removed} chunk(s) supprimé(s)")

        total = len(chunks)
        n_batches = (total + self.CHROMA_BATCH_SIZE - 1) // self.CHROMA_BATCH_SIZE
        logger.info(f"Indexation de {total} chunks en {n_batches} batch(es) dans ChromaDB...")

        # Premier batch : crée la collection
        first_batch = chunks[:self.CHROMA_BATCH_SIZE]
        self._vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        logger.info(f"  Batch 1/{n_batches} : {len(first_batch)} chunks indexés")

        # Batches suivants : ajout incrémental
        for i in range(1, n_batches):
            start = i * self.CHROMA_BATCH_SIZE
            batch = chunks[start:start + self.CHROMA_BATCH_SIZE]
            self._vectorstore.add_documents(batch)
            logger.info(f"  Batch {i+1}/{n_batches} : {len(batch)} chunks indexés")

        count = self._vectorstore._collection.count()
        logger.info(f"Base vectorielle créée : {count} vecteurs dans {self.persist_dir}")
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

        # MMR (Maximum Marginal Relevance) : diversifie les chunks récupérés
        # pour éviter que tous viennent du même paragraphe/section.
        # fetch_k = candidats analysés, k = retenus après diversification.
        effective_k = k or settings.RETRIEVER_K
        return self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": effective_k, "fetch_k": effective_k * 4},
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
