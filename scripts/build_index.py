# ── scripts/build_index.py ───────────────────────────────────────────────────
# Script à lancer UNE SEULE FOIS pour indexer les documents dans ChromaDB.
# À relancer si vous ajoutez de nouveaux documents dans data/docs/.
#
# Utilisation :
#   python scripts/build_index.py
#   python scripts/build_index.py --docs-dir /chemin/vers/docs
#   python scripts/build_index.py --reset   # Réindexe tout depuis zéro

import sys
import argparse
import sqlite3
import time
from pathlib import Path

# Racine du projet dans le path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.deduplication import deduplicate_chunks
from src.ingestion.vectorstore import VectorStoreManager


def _fix_chromadb_schema(persist_dir: str):
    """Ajoute la colonne 'topic' manquante après chaque build (bug ChromaDB 0.4.24)."""
    db_path = Path(persist_dir) / "chroma.sqlite3"
    if not db_path.exists():
        return
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    for table in ("collections", "segments"):
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        if "topic" not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN topic TEXT")
            logger.debug(f"ChromaDB schema fix : colonne 'topic' ajoutée à {table}")
    con.commit()
    con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Indexation des documents dans ChromaDB"
    )
    parser.add_argument(
        "--docs-dir",
        default=settings.DOCS_DIR,
        help=f"Dossier contenant les documents (défaut: {settings.DOCS_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprimer et recréer la base vectorielle complète",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.92,
        help="Seuil de déduplication des chunks [0,1] (défaut: 0.92)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("INDEXATION DES DOCUMENTS — Phase 1 RAG Chatbot")
    logger.info("=" * 60)
    logger.info(f"Dossier source  : {args.docs_dir}")
    logger.info(f"Base ChromaDB   : {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"Collection      : {settings.CHROMA_COLLECTION}")
    logger.info(f"Chunk size      : {settings.CHUNK_SIZE} tokens")
    logger.info(f"Chunk overlap   : {settings.CHUNK_OVERLAP} tokens")
    logger.info(f"Embeddings      : {settings.EMBEDDING_PROVIDER} / {settings.EMBEDDING_MODEL}")
    logger.info("-" * 60)

    # Optionnel : supprimer l'ancienne base
    if args.reset:
        import shutil
        if Path(settings.CHROMA_PERSIST_DIR).exists():
            shutil.rmtree(settings.CHROMA_PERSIST_DIR)
            logger.warning("Ancienne base ChromaDB supprimée (--reset)")

    start = time.time()

    # Étape 1 : Chargement et chunking
    logger.info("Étape 1/2 : Chargement et découpage des documents...")
    loader = DocumentLoader()
    chunks = loader.load_and_split(args.docs_dir)

    if not chunks:
        logger.error(
            f"Aucun document trouvé dans {args.docs_dir}.\n"
            "Placer des fichiers PDF/DOCX/TXT/MD dans ce dossier."
        )
        sys.exit(1)

    # Étape 1 bis : déduplication locale des chunks avant vectorisation
    before_dedup = len(chunks)
    chunks = deduplicate_chunks(chunks, similarity_threshold=args.dedupe_threshold)
    removed = before_dedup - len(chunks)
    if removed > 0:
        logger.info(f"Déduplication : {removed} chunk(s) quasi-identiques supprimé(s)")

    # Étape 2 : Vectorisation et stockage
    logger.info("Étape 2/2 : Vectorisation et stockage dans ChromaDB...")
    logger.info("(Cette étape peut prendre quelques minutes selon la taille des docs)")

    vs_manager = VectorStoreManager()
    vs_manager.build(chunks)

    # ChromaDB 0.4.24 bug : migration 5 supprime 'topic' mais le code le requête encore.
    # On le réajoute après chaque build pour éviter l'erreur au chargement.
    _fix_chromadb_schema(settings.CHROMA_PERSIST_DIR)

    elapsed = time.time() - start
    logger.info("-" * 60)
    logger.info(f"Indexation terminée en {elapsed:.1f}s")
    logger.info(f"{len(chunks)} chunks indexés dans {settings.CHROMA_PERSIST_DIR}")
    logger.info("")
    logger.info("Lancer le chatbot avec :")
    logger.info("  streamlit run src/interface/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
