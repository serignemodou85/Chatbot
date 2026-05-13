"""
Évaluation de la qualité du pipeline RAG.

Métriques mesurées :
  1. keyword_ratio     — mots-clés attendus trouvés dans la réponse
  2. source_relevance  — les sources citées sont-elles pertinentes au sujet ?
  3. has_answer        — la réponse contient-elle du contenu réel ?
  4. latency_s         — temps de réponse en secondes
  5. sources_count     — nombre de sources distinctes citées

Usage :
  python scripts/evaluate_rag.py
  python scripts/evaluate_rag.py --target 0.8 --verbose
  python scripts/evaluate_rag.py --mode "🔒 Pro cybersécurité"
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

# Ajouter la racine du projet au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.rag_chain import RAGChain


# ── Métriques ─────────────────────────────────────────────────────────────────

def keyword_score(answer: str, expected_keywords: List[str]) -> Tuple[float, List[str]]:
    """Retourne le ratio de mots-clés trouvés et la liste des mots trouvés."""
    if not expected_keywords:
        return 0.0, []
    lowered = answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in lowered]
    return len(hits) / len(expected_keywords), hits


def source_relevance_score(sources: List[Dict], topic_keywords: List[str]) -> Tuple[float, List[str]]:
    """
    Vérifie que les sources citées sont pertinentes au sujet de la question.
    Compare les noms de fichiers sources avec les mots-clés du sujet.

    Ex: si topic_keywords=["wazuh"], et source vient de "wazuh_agent.pdf" → pertinent.
    """
    if not sources or not topic_keywords:
        return 0.0, []

    relevant_sources = []
    for src in sources:
        filepath = src.get("file", "").lower()
        filename = Path(filepath).name.lower()
        # Un fichier source est considéré pertinent si son nom contient un mot-clé
        if any(kw.lower() in filename for kw in topic_keywords):
            relevant_sources.append(Path(filepath).name)

    ratio = len(relevant_sources) / len(sources) if sources else 0.0
    return ratio, list(set(relevant_sources))


def has_substantive_answer(answer: str, min_length: int = 50) -> bool:
    """La réponse contient-elle du contenu réel (pas juste un refus) ?"""
    if len(answer.strip()) < min_length:
        return False
    refusal_phrases = [
        "je ne sais pas",
        "pas dans le contexte",
        "je n'ai pas d'information",
        "i don't know",
        "not in the context",
    ]
    return not any(phrase in answer.lower() for phrase in refusal_phrases)


# ── Évaluation d'un cas ────────────────────────────────────────────────────────

def evaluate_case(chain: RAGChain, case: Dict, user_mode: str) -> Dict:
    question = case.get("question", "").strip()
    expected_keywords = case.get("expected_keywords", [])
    min_ratio = float(case.get("min_keyword_ratio", 0.6))
    # Mots-clés de sujet pour la pertinence des sources (premier mot-clé = nom du produit)
    topic_keywords = case.get("topic_keywords", expected_keywords[:2])

    if not question:
        raise ValueError("Chaque cas doit contenir un champ 'question' non vide.")

    # ── Appel RAG avec mesure du temps ────────────────────────────────────────
    t0 = time.perf_counter()
    response = chain.ask(question, user_mode=user_mode)
    latency = round(time.perf_counter() - t0, 2)

    answer = response.get("answer", "")
    sources = response.get("sources", [])

    # ── Calcul des métriques ──────────────────────────────────────────────────
    kw_ratio, kw_hits = keyword_score(answer, expected_keywords)
    src_ratio, relevant_src = source_relevance_score(sources, topic_keywords)
    substantive = has_substantive_answer(answer)

    # Pertinence globale : keyword_ratio >= min_ratio ET réponse substantielle
    relevant = (kw_ratio >= min_ratio) and substantive

    return {
        "question": question,
        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
        "metrics": {
            "keyword_ratio": round(kw_ratio, 3),
            "keyword_hits": kw_hits,
            "expected_keywords": expected_keywords,
            "min_keyword_ratio": min_ratio,
            "source_relevance_ratio": round(src_ratio, 3),
            "relevant_sources": relevant_src,
            "sources_count": len(sources),
            "has_substantive_answer": substantive,
            "latency_s": latency,
        },
        "relevant": relevant,
    }


# ── Rapport console ────────────────────────────────────────────────────────────

def print_report(summary: Dict, details: List[Dict], verbose: bool = False):
    """Affiche un rapport lisible dans le terminal."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  RAPPORT D'ÉVALUATION RAG")
    print(sep)
    print(f"  Questions testées : {summary['total_questions']}")
    print(f"  Réponses pertinentes : {summary['relevant_answers']}/{summary['total_questions']}")
    print(f"  Taux de pertinence : {summary['relevance_rate']:.1%}")
    print(f"  Latence moyenne : {summary['avg_latency_s']:.2f}s")
    print(f"  Objectif ({summary['target']:.0%}) atteint : {'✓ OUI' if summary['target_reached'] else '✗ NON'}")
    print(sep)

    for i, item in enumerate(details, 1):
        m = item["metrics"]
        status = "✓" if item["relevant"] else "✗"
        print(f"\n  [{status}] Q{i}: {item['question'][:65]}")
        print(f"       Mots-clés  : {m['keyword_ratio']:.0%} ({len(m['keyword_hits'])}/{len(m['expected_keywords'])} — trouvés: {m['keyword_hits']})")
        print(f"       Sources    : {m['sources_count']} citées, pertinence: {m['source_relevance_ratio']:.0%} ({m['relevant_sources']})")
        print(f"       Substantif : {'oui' if m['has_substantive_answer'] else 'NON ← à investiguer'}")
        print(f"       Latence    : {m['latency_s']}s")
        if verbose:
            print(f"\n       Réponse:\n       {item['answer_preview']}\n")

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Évalue la qualité du chatbot RAG sur un jeu de questions."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/test_questions.json",
        help="Chemin du dataset JSON d'évaluation.",
    )
    parser.add_argument(
        "--mode",
        default="🖥️ Admin système",
        choices=["🎓 Étudiant", "🖥️ Admin système", "🔒 Pro cybersécurité"],
        help="Mode utilisateur utilisé pendant l'évaluation.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.8,
        help="Objectif minimal de pertinence (défaut: 0.8)",
    )
    parser.add_argument(
        "--report",
        default="data/eval/last_report.json",
        help="Chemin du rapport JSON de sortie.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les réponses dans le rapport console.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    report_path = Path(args.report)

    logger.info("Chargement du jeu d'évaluation...")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if not eval_set:
        raise ValueError("Le dataset d'évaluation est vide.")

    logger.info("Initialisation du pipeline RAG...")
    chain = RAGChain()

    logger.info(f"Évaluation de {len(eval_set)} questions en mode '{args.mode}'...")
    details = []
    for case in eval_set:
        result = evaluate_case(chain, case, args.mode)
        status = "✓" if result["relevant"] else "✗"
        logger.info(f"  [{status}] {case['question'][:60]} — {result['metrics']['latency_s']}s")
        details.append(result)

    relevant_count = sum(1 for item in details if item["relevant"])
    relevance_rate = relevant_count / len(details)
    avg_latency = round(sum(d["metrics"]["latency_s"] for d in details) / len(details), 2)

    summary = {
        "total_questions": len(details),
        "relevant_answers": relevant_count,
        "relevance_rate": round(relevance_rate, 3),
        "avg_latency_s": avg_latency,
        "target": args.target,
        "target_reached": relevance_rate >= args.target,
        "mode": args.mode,
    }

    report = {"summary": summary, "details": details}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(summary, details, verbose=args.verbose)
    logger.info(f"Rapport sauvegardé : {report_path}")

    # Retourner exit code 1 si l'objectif n'est pas atteint (utile pour CI/CD)
    sys.exit(0 if summary["target_reached"] else 1)


if __name__ == "__main__":
    main()
