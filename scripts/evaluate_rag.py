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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Windows cp1252 → forcer UTF-8 pour les caractères spéciaux (✓ ✗ é...)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
        # Vérifier chemin complet (inclut le dossier ex: data/docs/zabbix/...)
        # car certains fichiers n'ont pas le nom du produit dans leur titre
        if any(kw.lower() in filepath for kw in topic_keywords):
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


# ── Rapport HTML ──────────────────────────────────────────────────────────────

def generate_html_report(summary: Dict, details: List[Dict]) -> str:
    """Génère un rapport HTML autonome (CSS inline, imprimable en PDF)."""
    date_str = datetime.now().strftime("%d/%m/%Y à %Hh%M")
    target_pct = f"{summary['target']:.0%}"
    rate_pct   = f"{summary['relevance_rate']:.1%}"
    target_ok  = summary["target_reached"]
    badge_color = "#16a34a" if target_ok else "#dc2626"
    badge_text  = "✓ Objectif atteint" if target_ok else "✗ Objectif non atteint"

    rows = ""
    for i, item in enumerate(details, 1):
        m      = item["metrics"]
        status = "✓" if item["relevant"] else "✗"
        color  = "#16a34a" if item["relevant"] else "#dc2626"
        kw_pct = f"{m['keyword_ratio']:.0%}"
        src_pct = f"{m['source_relevance_ratio']:.0%}"
        answer_html = item["answer_preview"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        rows += f"""
        <tr>
          <td style="color:{color};font-weight:bold;text-align:center">{status}</td>
          <td>Q{i} — {item['question']}</td>
          <td style="text-align:center">{kw_pct}<br><small style="color:#6b7280">{m['keyword_hits']}</small></td>
          <td style="text-align:center">{src_pct}</td>
          <td style="text-align:center">{m['latency_s']}s</td>
        </tr>
        <tr>
          <td></td>
          <td colspan="4" style="background:#f8fafc;padding:8px 12px;font-size:0.85em;color:#374151;border-left:3px solid #e5e7eb">
            {answer_html}
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rapport d'évaluation RAG — InfraBot</title>
<style>
  @media print {{ body {{ font-size: 11pt; }} .no-print {{ display:none; }} }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 40px auto;
         padding: 0 24px; color: #111827; background: #fff; }}
  h1   {{ font-size: 1.6em; margin-bottom: 4px; }}
  .sub {{ color: #6b7280; font-size: 0.9em; margin-bottom: 32px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
  .card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
           padding: 16px 20px; min-width: 160px; flex: 1; }}
  .card .val {{ font-size: 2em; font-weight: 700; }}
  .card .lbl {{ font-size: 0.82em; color: #6b7280; margin-top: 2px; }}
  .badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px;
            color: #fff; font-weight: 600; background: {badge_color}; margin-bottom: 28px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th    {{ background: #1e3a5f; color: #fff; padding: 10px 14px; text-align: left; }}
  td    {{ padding: 8px 14px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
  tr:hover td {{ background: #f0f9ff; }}
  .footer {{ margin-top: 40px; font-size: 0.8em; color: #9ca3af; text-align: center; }}
</style>
</head>
<body>
<h1>🛡️ InfraBot — Rapport d'évaluation RAG</h1>
<div class="sub">Généré le {date_str} · Mode : {summary['mode']} · Objectif : {target_pct}</div>

<div class="cards">
  <div class="card">
    <div class="val">{summary['relevant_answers']}/{summary['total_questions']}</div>
    <div class="lbl">Questions pertinentes</div>
  </div>
  <div class="card">
    <div class="val">{rate_pct}</div>
    <div class="lbl">Taux de pertinence</div>
  </div>
  <div class="card">
    <div class="val">{summary['avg_latency_s']:.1f}s</div>
    <div class="lbl">Latence moyenne</div>
  </div>
</div>

<div class="badge">{badge_text}</div>

<table>
  <thead>
    <tr>
      <th style="width:40px"></th>
      <th>Question</th>
      <th style="width:110px">Mots-clés</th>
      <th style="width:90px">Sources</th>
      <th style="width:70px">Latence</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<div class="footer">InfraBot RAG Chatbot · Phase 1 + Phase 2 · LangChain + ChromaDB + llama3.2 + CrewAI</div>
</body>
</html>"""


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

    dataset_path = Path(args.dataset).resolve()
    report_path  = Path(args.report).resolve()
    project_root = Path(__file__).parent.parent.resolve()

    # Restreindre la lecture/écriture au dossier du projet
    if not str(dataset_path).startswith(str(project_root)):
        raise ValueError(f"--dataset doit être dans le projet : {dataset_path}")
    if not str(report_path).startswith(str(project_root)):
        raise ValueError(f"--report doit être dans le projet : {report_path}")

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
        # Réinitialiser la mémoire entre chaque question pour une évaluation indépendante.
        # En production, chaque session Streamlit démarre avec une mémoire vide.
        chain.reset_memory()
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

    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Rapport HTML (principal — lisible + imprimable en PDF)
    html_path = report_path.with_suffix(".html")
    html_content = generate_html_report(summary, details)
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Rapport HTML sauvegardé : {html_path}")

    print_report(summary, details, verbose=args.verbose)

    # Retourner exit code 1 si l'objectif n'est pas atteint (utile pour CI/CD)
    sys.exit(0 if summary["target_reached"] else 1)


if __name__ == "__main__":
    main()
