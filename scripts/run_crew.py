"""Test CLI du crew Phase 2.

Usage :
    $env:PYTHONPATH = "."
    venv\\Scripts\\python scripts/run_crew.py
    venv\\Scripts\\python scripts/run_crew.py --mode admin
    venv\\Scripts\\python scripts/run_crew.py --question "Compare Wazuh et Zabbix"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.crew import CyberSecCrew, _classify

MODES = {
    "etudiant": "\U0001f393 Étudiant",
    "admin":    "\U0001f5a5️ Admin système",
    "pro":      "\U0001f512 Pro cybersécurité",
}

DEFAULT_QUESTIONS = [
    ("Compare Wazuh et Zabbix pour la supervision d'un réseau de 50 machines", "admin"),
    ("Comment configurer un VPN site-à-site avec pfSense ?",                    "admin"),
    ("Quels sont les risques de sécurité d'un serveur Ubuntu sans pare-feu ?",  "pro"),
]


def main():
    parser = argparse.ArgumentParser(description="Test CLI du crew Phase 2")
    parser.add_argument("--question", type=str, default=None, help="Question à poser")
    parser.add_argument("--mode",     type=str, default="admin",
                        choices=["etudiant", "admin", "pro"])
    parser.add_argument("--classify-only", action="store_true",
                        help="Affiche seulement le routage, sans lancer le crew")
    args = parser.parse_args()

    user_mode = MODES[args.mode]
    question  = args.question or DEFAULT_QUESTIONS[0][0]

    def safe(s: str) -> str:
        return s.encode("cp1252", "replace").decode("cp1252")

    print(f"\n{'='*60}")
    print(f"  CREW PHASE 2 - TEST CLI")
    print(f"{'='*60}")
    print(f"  Question : {safe(question)}")
    print(f"  Mode     : {safe(user_mode)}")
    domains = _classify(question)
    print(f"  Routage  : {domains}")
    print(f"{'='*60}\n")

    if args.classify_only:
        return

    crew = CyberSecCrew()
    t0 = time.time()
    result = crew.run(question, user_mode)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  RAPPORT FINAL  ({elapsed:.1f}s)")
    print(f"  Agents utilises : {safe(str(result['agents_used']))}")
    print(f"{'='*60}")
    report = safe(result["report"])
    print(report)


if __name__ == "__main__":
    main()
