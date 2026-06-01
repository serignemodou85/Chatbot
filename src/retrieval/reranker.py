# ── src/retrieval/reranker.py ─────────────────────────────────────────────────
# Cross-encoder re-ranking : prend N candidats issus de la recherche vectorielle
# et les réordonne par score de pertinence croisée (modèle multilingue FR/EN).
#
# Pipeline :
#   similarity_search(k=fetch_k) → candidates
#   CrossEncoder.predict(query, each_candidate) → scores
#   sort by score desc → top_k docs
#
# Modèle : cross-encoder/ms-marco-multilingual-MiniLM-L12-v2
#   ~500 MB, chargé une seule fois (singleton lru_cache), supporte FR + EN.

from __future__ import annotations

from functools import lru_cache
from typing import Any, List

from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from loguru import logger

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

_encoder_available: bool | None = None  # None = not yet tested


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder
    logger.info(f"Chargement du re-ranker : {_MODEL_NAME}")
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, docs: List[Document], top_k: int = 6) -> List[Document]:
    """
    Re-classe docs par pertinence croisée avec query.
    Retourne les top_k documents dans l'ordre décroissant de score.
    Si le modèle ne peut pas être chargé, retourne les top_k premiers sans re-ranking.
    """
    global _encoder_available
    if len(docs) <= 1:
        return docs[:top_k]
    if _encoder_available is False:
        return docs[:top_k]
    try:
        encoder = _load_cross_encoder()
        _encoder_available = True
        pairs = [(query, doc.page_content) for doc in docs]
        scores = encoder.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
        result = [doc for _, doc in ranked[:top_k]]
        logger.debug(f"[Re-ranker] {len(docs)} candidats → top {len(result)}")
        return result
    except Exception as e:
        _encoder_available = False
        logger.warning(f"[Re-ranker] Modèle indisponible, fallback sans re-ranking : {e}")
        return docs[:top_k]


class RerankedRetriever(BaseRetriever):
    """
    Retriever LangChain qui remplace MMR par similarity_search + cross-encoder.

    fetch_k : nombre de candidats récupérés depuis ChromaDB
    top_k   : nombre de documents retenus après re-ranking
    """

    vsm: Any
    fetch_k: int = 24
    top_k: int = 6

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        if self.vsm._vectorstore is None:
            self.vsm.load()
        candidates = self.vsm._vectorstore.similarity_search(query, k=self.fetch_k)
        return rerank(query, candidates, top_k=self.top_k)
