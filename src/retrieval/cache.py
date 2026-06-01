# ── src/retrieval/cache.py ────────────────────────────────────────────────────
# Cache sémantique LRU pour les réponses RAG et Crew.
#
# Principe : deux questions avec cosine(emb_q1, emb_q2) ≥ THRESHOLD et
# le même user_mode retournent le résultat mis en cache sans relancer le LLM.
#
# Embeddings : réutilise le modèle HuggingFace déjà chargé en mémoire
# (singleton _get_vsm() de rag_tools.py) — pas de double chargement.
#
# Éviction : LRU, limite MAX_ENTRIES (200 par défaut).
# Seuil    : 0.95 (intentionnellement élevé — RAG cybersec = précision requise).

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np
from loguru import logger

MAX_ENTRIES = 200
SIMILARITY_THRESHOLD = 0.95


class SemanticCache:
    """
    Cache sémantique LRU.

    get(question, user_mode) → résultat si une question similaire est en cache, sinon None.
    set(question, user_mode, result) → stocke le résultat.
    clear() → vide le cache.
    stats → dict hits/misses/hit_rate.
    """

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        # OrderedDict pour l'éviction LRU (popitem last=False)
        # value = (embedding, user_mode, result, timestamp)
        self._store: OrderedDict[str, tuple[np.ndarray, str, Any, float]] = OrderedDict()
        self._max = max_entries
        self._threshold = threshold
        self._hits = 0
        self._misses = 0

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        from src.agents.tools.rag_tools import _get_vsm
        emb = _get_vsm().embedding_model.embed_query(text)
        return np.array(emb, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / denom)

    # ── API publique ───────────────────────────────────────────────────────────

    def get(self, question: str, user_mode: str) -> Any | None:
        if not self._store:
            self._misses += 1
            return None
        q_emb = self._embed(question)
        best_sim, best_result = 0.0, None
        for _key, (emb, mode, result, _ts) in self._store.items():
            if mode != user_mode:
                continue
            sim = self._cosine(q_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_result = result
        if best_result is not None and best_sim >= self._threshold:
            self._hits += 1
            logger.info(f"[Cache] HIT — cosine={best_sim:.3f}")
            return best_result
        self._misses += 1
        return None

    def set(self, question: str, user_mode: str, result: Any) -> None:
        q_emb = self._embed(question)
        if len(self._store) >= self._max:
            self._store.popitem(last=False)
        self._store[question] = (q_emb, user_mode, result, time.time())
        logger.debug(f"[Cache] SET — {len(self._store)}/{self._max} entrées")

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
        logger.info("[Cache] Vidé.")

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 2),
        }
