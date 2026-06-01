"""
Test des 15 questions de validation — 3 niveaux x 5 questions.
Lance avec : python scripts/test_questions.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.rag_chain import RAGChain

QUESTIONS = [
    # ── Niveau Étudiant ──────────────────────────────────────────────────────
    ("🎓 Étudiant", "Q1",  "À quoi sert une règle firewall sur pfSense et comment autoriser un service comme SSH ?"),
    ("🎓 Étudiant", "Q2",  "Qu'est-ce qu'un tunnel VPN IPSec et à quoi sert la phase 1 ?"),
    ("🎓 Étudiant", "Q3",  "Comment afficher les logs SSH sur Linux pour voir les tentatives de connexion ?"),
    ("🎓 Étudiant", "Q4",  "Qu'est-ce qu'un utilisateur Active Directory dans Windows Server ?"),
    ("🎓 Étudiant", "Q5",  "À quoi sert un agent Zabbix sur une machine Linux ?"),
    # ── Niveau Admin ─────────────────────────────────────────────────────────
    ("🖥️ Admin système", "Q6",  "Comment créer une règle pfSense pour autoriser SSH uniquement depuis un réseau spécifique ?"),
    ("🖥️ Admin système", "Q7",  "Comment diagnostiquer un échec de connexion SSH avec journalctl et sshd_config ?"),
    ("🖥️ Admin système", "Q8",  "Comment configurer un agent Zabbix pour surveiller un serveur Linux et remonter les métriques CPU/RAM ?"),
    ("🖥️ Admin système", "Q9",  "Comment analyser un échec d'authentification dans Windows Server avec les événements 4625 et 4624 ?"),
    ("🖥️ Admin système", "Q10", "Comment vérifier la configuration d'un tunnel IPSec strongSwan avec swanctl --list-conns ?"),
    # ── Niveau Pro ───────────────────────────────────────────────────────────
    ("🔒 Pro cybersécurité", "Q11", "Comment corréler une attaque brute force SSH dans Wazuh avec la technique MITRE ATT&CK T1110 ?"),
    ("🔒 Pro cybersécurité", "Q12", "Comment interpréter l'erreur strongSwan NO_PROPOSAL_CHOSEN dans les logs charon et identifier le paramètre cryptographique en conflit ?"),
    ("🔒 Pro cybersécurité", "Q13", "Comment détecter une élévation de privilèges sur Windows via les événements 4672 et 4732 ?"),
    ("🔒 Pro cybersécurité", "Q14", "Comment analyser un trafic réseau suspect sur Linux avec ss, ip et tcpdump pour identifier un C2 potentiel ?"),
    ("🔒 Pro cybersécurité", "Q15", "Comment configurer Wazuh FIM pour surveiller les modifications non autorisées de fichiers critiques du système ?"),
]

SEP = "─" * 70

def evaluate_answer(answer: str, sources: list) -> dict:
    """Évalue rapidement la qualité d'une réponse."""
    blocked   = "Information absente" in answer or "absente du contexte" in answer
    gui_only  = "GUI uniquement" in answer or "interface graphique" in answer.lower()
    has_cmd   = any(c in answer for c in ["```", "`", "$ ", "# "])
    n_sources = len(sources)
    length    = len(answer)

    if blocked:
        status = "⛔ BLOQUÉ (validator)"
    elif gui_only:
        status = "🖥️ GUI only (pfSense correct)"
    elif length < 100:
        status = "⚠️  Réponse trop courte"
    elif n_sources == 0:
        status = "⚠️  Aucune source"
    else:
        status = "✅ OK"

    return {
        "status": status,
        "blocked": blocked,
        "has_cmd": has_cmd,
        "n_sources": n_sources,
        "length": length,
    }


def main():
    print(f"\n{'='*70}")
    print("  TEST 15 QUESTIONS — InfraBot RAG")
    print(f"{'='*70}\n")

    print("Chargement du pipeline RAG...")
    chain = RAGChain()
    print("Pipeline prêt.\n")

    results = []

    for mode, qid, question in QUESTIONS:
        print(f"{SEP}")
        print(f"[{qid}] {mode}")
        print(f"Q: {question}")
        print()

        t0 = time.time()
        try:
            result  = chain.ask(question, user_mode=mode)
            elapsed = time.time() - t0
            answer  = result["answer"]
            sources = result.get("sources", [])

            ev = evaluate_answer(answer, sources)
            print(f"Status  : {ev['status']}")
            print(f"Durée   : {elapsed:.1f}s  |  Longueur : {ev['length']} chars  |  Sources : {ev['n_sources']}")

            # Aperçu réponse
            preview = answer[:300].replace("\n", " ")
            print(f"Réponse : {preview}{'...' if len(answer) > 300 else ''}")

            if sources:
                for s in sources[:2]:
                    print(f"  📄 {Path(s['file']).name} p.{s.get('page','?')}")

            results.append({
                "qid": qid, "mode": mode, "question": question,
                "status": ev["status"], "blocked": ev["blocked"],
                "sources": ev["n_sources"], "length": ev["length"],
            })

        except Exception as e:
            print(f"ERREUR: {e}")
            results.append({"qid": qid, "mode": mode, "blocked": True, "status": f"❌ ERREUR: {e}"})

        chain.reset_memory()  # isolation entre les questions
        print()

    # Résumé
    print(f"\n{'='*70}")
    print("  RÉSUMÉ")
    print(f"{'='*70}")
    ok      = sum(1 for r in results if "✅" in r.get("status",""))
    blocked = sum(1 for r in results if r.get("blocked"))
    warn    = sum(1 for r in results if "⚠️" in r.get("status",""))
    print(f"  ✅ OK          : {ok}/15")
    print(f"  ⛔ Bloquées    : {blocked}/15")
    print(f"  ⚠️  Warnings   : {warn}/15")
    print()
    print("  Détail par question :")
    for r in results:
        src = f"({r.get('sources',0)} sources)" if not r.get("blocked") else ""
        print(f"  [{r['qid']:3}] {r['status']} {src}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
