from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

from langchain.schema import Document


def normalize_text(text: str) -> str:
    """Normalize text to make near-duplicate detection more robust."""
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _length_bucket(size: int, bucket_width: int = 100) -> int:
    return size // bucket_width


def similarity(a: str, b: str) -> float:
    """Character-level similarity ratio in [0, 1]."""
    return SequenceMatcher(None, a, b).ratio()


def find_duplicate_ids(
    texts_by_id: Dict[str, str],
    similarity_threshold: float = 0.92,
) -> List[str]:
    """
    Return ids to drop by keeping the first representative of each near-duplicate group.
    """
    buckets: Dict[int, List[Tuple[str, str]]] = {}
    duplicate_ids: List[str] = []

    for item_id, raw_text in texts_by_id.items():
        text = normalize_text(raw_text)
        if not text:
            duplicate_ids.append(item_id)
            continue

        bucket = _length_bucket(len(text))
        candidate_buckets = (bucket - 1, bucket, bucket + 1)

        is_duplicate = False
        for candidate_bucket in candidate_buckets:
            for existing_id, existing_text in buckets.get(candidate_bucket, []):
                if similarity(text, existing_text) >= similarity_threshold:
                    duplicate_ids.append(item_id)
                    is_duplicate = True
                    break
            if is_duplicate:
                break

        if not is_duplicate:
            buckets.setdefault(bucket, []).append((item_id, text))

    return duplicate_ids


MAX_CHUNKS_FOR_DEDUP = 3000


def deduplicate_chunks(
    chunks: Iterable[Document],
    similarity_threshold: float = 0.92,
) -> List[Document]:
    """
    Remove near-identical chunks while preserving original order.

    Auto-skip si le corpus dépasse MAX_CHUNKS_FOR_DEDUP (3000) :
    SequenceMatcher est O(n²) dans les buckets — au-delà de 3000 chunks
    le temps de calcul devient prohibitif (>30 min sur CPU).
    Utiliser deduplicate_chroma.py en post-traitement dans ce cas.
    """
    chunk_list = list(chunks)

    if len(chunk_list) > MAX_CHUNKS_FOR_DEDUP:
        from loguru import logger as _log
        _log.warning(
            f"Déduplication ignorée : {len(chunk_list)} chunks > limite {MAX_CHUNKS_FOR_DEDUP}. "
            "Utiliser deduplicate_chroma.py en post-traitement."
        )
        return chunk_list

    texts_by_id: Dict[str, str] = {}
    index_to_doc: Dict[str, Document] = {}

    for idx, chunk in enumerate(chunk_list):
        item_id = f"chunk-{idx}"
        texts_by_id[item_id] = chunk.page_content
        index_to_doc[item_id] = chunk

    to_drop = set(find_duplicate_ids(texts_by_id, similarity_threshold=similarity_threshold))
    return [
        index_to_doc[f"chunk-{idx}"]
        for idx in range(len(texts_by_id))
        if f"chunk-{idx}" not in to_drop
    ]
