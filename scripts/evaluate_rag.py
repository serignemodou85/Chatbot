import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from src.retrieval.rag_chain import RAGChain


def load_eval_set(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier d'evaluation introuvable: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Le jeu d'evaluation doit etre une liste JSON d'objets.")
    return data


def keyword_score(answer: str, expected_keywords: List[str]) -> Tuple[float, int]:
    if not expected_keywords:
        return 0.0, 0
    lowered = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return hits / len(expected_keywords), hits


def evaluate_case(chain: RAGChain, case: Dict, user_mode: str) -> Dict:
    question = case.get("question", "").strip()
    expected_keywords = case.get("expected_keywords", [])
    min_ratio = float(case.get("min_keyword_ratio", 0.6))

    if not question:
        raise ValueError("Chaque cas doit contenir un champ 'question' non vide.")
    if not isinstance(expected_keywords, list):
        raise ValueError("Le champ 'expected_keywords' doit etre une liste.")

    response = chain.ask(question, user_mode=user_mode)
    answer = response.get("answer", "")
    ratio, hits = keyword_score(answer, expected_keywords)
    relevant = ratio >= min_ratio

    return {
        "question": question,
        "answer": answer,
        "expected_keywords": expected_keywords,
        "keyword_hits": hits,
        "keyword_ratio": round(ratio, 3),
        "min_keyword_ratio": min_ratio,
        "relevant": relevant,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evalue la pertinence du chatbot RAG sur un jeu de questions."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/test_questions.json",
        help="Chemin du dataset JSON d'evaluation.",
    )
    parser.add_argument(
        "--mode",
        default=" Admin système",
        choices=["🎓 Étudiant", "🖥️ Admin système", "🔒 Pro cybersécurité"],
        help="Mode utilisateur utilise pendant l'evaluation.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.8,
        help="Objectif minimal de pertinence (defaut: 0.8)",
    )
    parser.add_argument(
        "--report",
        default="data/eval/last_report.json",
        help="Chemin du rapport JSON de sortie.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    report_path = Path(args.report)

    logger.info("Chargement du jeu d'evaluation...")
    eval_set = load_eval_set(dataset_path)
    if not eval_set:
        raise ValueError("Le dataset d'evaluation est vide.")

    logger.info("Initialisation du pipeline RAG...")
    chain = RAGChain()

    logger.info(f"Evaluation de {len(eval_set)} questions en mode '{args.mode}'...")
    details = [evaluate_case(chain, case, args.mode) for case in eval_set]
    relevant_count = sum(1 for item in details if item["relevant"])
    relevance_rate = relevant_count / len(details)

    summary = {
        "total_questions": len(details),
        "relevant_answers": relevant_count,
        "relevance_rate": round(relevance_rate, 3),
        "target": args.target,
        "target_reached": relevance_rate >= args.target,
        "mode": args.mode,
    }

    report = {"summary": summary, "details": details}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("----- RESULTATS -----")
    logger.info(f"Pertinence: {summary['relevant_answers']}/{summary['total_questions']}")
    logger.info(f"Taux: {summary['relevance_rate']:.1%}")
    logger.info(f"Objectif {args.target:.0%} atteint: {summary['target_reached']}")
    logger.info(f"Rapport ecrit: {report_path}")


if __name__ == "__main__":
    main()
