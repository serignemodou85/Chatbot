import argparse
from pathlib import Path

import chromadb
from loguru import logger

from config.settings import settings
from src.ingestion.deduplication import find_duplicate_ids


def main():
    parser = argparse.ArgumentParser(
        description="Supprime les chunks quasi-identiques dans ChromaDB."
    )
    parser.add_argument(
        "--persist-dir",
        default=settings.CHROMA_PERSIST_DIR,
        help=f"Dossier ChromaDB (defaut: {settings.CHROMA_PERSIST_DIR})",
    )
    parser.add_argument(
        "--collection",
        default=settings.CHROMA_COLLECTION,
        help=f"Collection Chroma (defaut: {settings.CHROMA_COLLECTION})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Seuil de similarite [0,1] (defaut: 0.92)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les doublons detectes sans suppression.",
    )
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Repertoire Chroma introuvable: {persist_dir}\n"
            "Indexer d'abord avec: python scripts/build_index.py"
        )

    logger.info("Analyse de deduplication ChromaDB")
    logger.info(f"Persist dir : {persist_dir}")
    logger.info(f"Collection  : {args.collection}")
    logger.info(f"Threshold   : {args.threshold}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=args.collection)
    data = collection.get(include=["documents", "metadatas"])

    ids = data.get("ids", [])
    docs = data.get("documents", [])
    if not ids:
        logger.warning("Collection vide, aucune action.")
        return

    texts_by_id = {item_id: (doc or "") for item_id, doc in zip(ids, docs)}
    duplicate_ids = find_duplicate_ids(
        texts_by_id=texts_by_id,
        similarity_threshold=args.threshold,
    )

    logger.info(f"Total chunks en base : {len(ids)}")
    logger.info(f"Doublons detectes    : {len(duplicate_ids)}")

    if not duplicate_ids:
        logger.success("Aucun doublon quasi-identique detecte.")
        return

    if args.dry_run:
        logger.info("Mode dry-run: aucune suppression effectuee.")
        return

    confirm = input(
        f"\n⚠️  Suppression de {len(duplicate_ids)} chunks sur {len(ids)} "
        f"({len(duplicate_ids)/len(ids):.0%} de la base).\n"
        "Taper 'OUI' pour confirmer : "
    ).strip()
    if confirm != "OUI":
        logger.info("Opération annulée.")
        return

    collection.delete(ids=duplicate_ids)
    remaining = collection.count()
    logger.success(
        f"Suppression terminee: {len(duplicate_ids)} chunks supprimes, "
        f"{remaining} chunks restants."
    )


if __name__ == "__main__":
    main()
