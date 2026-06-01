"""
Évaluation RAG — 30 questions sur 3 niveaux et tous les domaines.
Lance avec : python scripts/evaluate_30.py
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retrieval.rag_chain import RAGChain

QUESTIONS = [
    # ═══════════════════════════════════════════════════════════════
    # 🎓  NIVEAU ÉTUDIANT — 10 questions
    # ═══════════════════════════════════════════════════════════════
    ("🎓 Étudiant", "Q01", "pfSense",
     "À quoi sert une règle firewall sur pfSense et comment en créer une pour autoriser SSH ?"),

    ("🎓 Étudiant", "Q02", "VPN / IPSec",
     "Qu'est-ce qu'un tunnel VPN IPSec et à quoi sert la phase 1 ?"),

    ("🎓 Étudiant", "Q03", "Linux / SSH",
     "Comment afficher les logs SSH sur Linux pour voir les tentatives de connexion ?"),

    ("🎓 Étudiant", "Q04", "Windows / AD",
     "Qu'est-ce qu'un utilisateur Active Directory dans Windows Server ?"),

    ("🎓 Étudiant", "Q05", "Zabbix",
     "À quoi sert un agent Zabbix installé sur une machine Linux ?"),

    ("🎓 Étudiant", "Q06", "Wazuh",
     "Qu'est-ce que Wazuh et quel est son rôle dans une architecture de sécurité ?"),

    ("🎓 Étudiant", "Q07", "Dolibarr",
     "Comment créer un utilisateur et lui attribuer des droits dans Dolibarr ?"),

    ("🎓 Étudiant", "Q08", "Wazuh / FIM",
     "Qu'est-ce que le File Integrity Monitoring (FIM) et pourquoi est-il important ?"),

    ("🎓 Étudiant", "Q09", "Linux / réseau",
     "À quoi sert la commande ip addr sur Linux et comment lire son résultat ?"),

    ("🎓 Étudiant", "Q10", "Windows / logs",
     "Qu'est-ce que l'événement Windows 4625 et que signifie-t-il ?"),

    # ═══════════════════════════════════════════════════════════════
    # 🖥️  NIVEAU ADMIN SYSTÈME — 10 questions
    # ═══════════════════════════════════════════════════════════════
    ("🖥️ Admin système", "Q11", "pfSense / firewall",
     "Comment créer une règle pfSense pour autoriser SSH uniquement depuis un réseau spécifique ?"),

    ("🖥️ Admin système", "Q12", "Linux / SSH",
     "Comment diagnostiquer un échec de connexion SSH avec journalctl et sshd_config ?"),

    ("🖥️ Admin système", "Q13", "Zabbix / agent",
     "Comment configurer un agent Zabbix pour surveiller un serveur Linux et remonter les métriques CPU et RAM ?"),

    ("🖥️ Admin système", "Q14", "Windows / audit",
     "Comment analyser un échec d'authentification dans Windows Server avec les événements 4625 et 4624 ?"),

    ("🖥️ Admin système", "Q15", "strongSwan / IPSec",
     "Comment vérifier la configuration d'un tunnel IPSec strongSwan avec swanctl --list-conns ?"),

    ("🖥️ Admin système", "Q16", "pfSense / OpenVPN",
     "Quelles sont les premières étapes pour configurer un serveur OpenVPN dans pfSense ?"),

    ("🖥️ Admin système", "Q17", "Wazuh / agent",
     "Comment installer et enregistrer un agent Wazuh sur un serveur Ubuntu ?"),

    ("🖥️ Admin système", "Q18", "Zabbix / trigger",
     "Comment créer un trigger Zabbix pour déclencher une alerte quand la CPU dépasse 90% ?"),

    ("🖥️ Admin système", "Q19", "Dolibarr / droits",
     "Comment configurer les droits et permissions d'un utilisateur dans Dolibarr ?"),

    ("🖥️ Admin système", "Q20", "Linux / systemd",
     "Comment diagnostiquer un service systemd en échec avec journalctl et systemctl ?"),

    # ═══════════════════════════════════════════════════════════════
    # 🔒  NIVEAU PRO CYBERSÉCURITÉ — 10 questions
    # ═══════════════════════════════════════════════════════════════
    ("🔒 Pro cybersécurité", "Q21", "Wazuh / MITRE",
     "Comment corréler une attaque brute force SSH dans Wazuh avec la technique MITRE ATT&CK T1110 ?"),

    ("🔒 Pro cybersécurité", "Q22", "strongSwan / logs",
     "Comment interpréter l'erreur NO_PROPOSAL_CHOSEN dans les logs charon de strongSwan et identifier le paramètre cryptographique en conflit ?"),

    ("🔒 Pro cybersécurité", "Q23", "Windows / privilege",
     "Comment détecter une élévation de privilèges sur Windows via les événements 4672 et 4732 ?"),

    ("🔒 Pro cybersécurité", "Q24", "Linux / C2",
     "Comment analyser un trafic réseau suspect sur Linux avec ss, ip et tcpdump pour identifier un potentiel C2 ?"),

    ("🔒 Pro cybersécurité", "Q25", "Wazuh / FIM",
     "Comment configurer Wazuh FIM pour surveiller les modifications non autorisées de fichiers critiques du système ?"),

    ("🔒 Pro cybersécurité", "Q26", "pfSense / logs",
     "Comment analyser les logs du firewall pfSense pour détecter un scan de ports ou une reconnaissance réseau ?"),

    ("🔒 Pro cybersécurité", "Q27", "Windows / RDP",
     "Comment détecter une attaque par force brute RDP via les événements Windows 4778 et 4625 ?"),

    ("🔒 Pro cybersécurité", "Q28", "MITRE ATT&CK",
     "Quelles sont les principales techniques MITRE ATT&CK associées à la tactique Persistence (TA0003) ?"),

    ("🔒 Pro cybersécurité", "Q29", "Wazuh / sudo",
     "Comment configurer les règles Wazuh pour détecter une utilisation anormale de sudo et une potentielle escalade de privilèges ?"),

    ("🔒 Pro cybersécurité", "Q30", "OpenVPN / logs",
     "Comment analyser les logs OpenVPN pour détecter une connexion non autorisée ou une anomalie d'authentification ?"),
]

SEP = "─" * 72


def classify(answer: str, sources: list) -> dict:
    blocked  = "Information absente" in answer or "GUI uniquement" in answer or "Incohérence" in answer or "Réponse insuffisante" in answer
    length   = len(answer.strip())
    n_src    = len(sources)
    too_short = not blocked and length < 100

    if blocked:
        status = "⛔ BLOQUÉ"
    elif too_short:
        status = "⚠️  COURT"
    elif n_src == 0:
        status = "⚠️  SANS SOURCE"
    else:
        status = "✅ OK"

    return {"status": status, "blocked": blocked, "short": too_short,
            "length": length, "sources": n_src}


def main():
    print(f"\n{'='*72}")
    print("  ÉVALUATION RAG — 30 QUESTIONS — InfraBot")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"{'='*72}\n")

    print("Chargement du pipeline RAG...")
    chain = RAGChain()
    print("Prêt.\n")

    results = []
    total_time = 0

    for mode, qid, domain, question in QUESTIONS:
        print(f"{SEP}")
        print(f"[{qid}] {mode}  |  {domain}")
        print(f"Q : {question}")

        t0 = time.time()
        try:
            r       = chain.ask(question, user_mode=mode)
            elapsed = time.time() - t0
            answer  = r["answer"]
            sources = r.get("sources", [])
            ev      = classify(answer, sources)

            total_time += elapsed
            print(f"     {ev['status']}  |  {elapsed:.0f}s  |  {ev['length']} chars  |  {ev['sources']} source(s)")

            preview = answer.replace("\n", " ")[:200]
            print(f"     {preview}{'...' if len(answer) > 200 else ''}")
            if sources:
                for s in sources[:2]:
                    print(f"     📄 {Path(s['file']).name}")

            results.append({**ev, "qid": qid, "mode": mode, "domain": domain,
                            "question": question, "latency": round(elapsed, 1),
                            "answer": answer, "sources_list": sources})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"     ❌ ERREUR : {e}")
            results.append({"status": "❌ ERREUR", "blocked": False, "short": False,
                            "length": 0, "sources": 0, "qid": qid, "mode": mode,
                            "domain": domain, "question": question, "latency": elapsed,
                            "answer": str(e), "sources_list": []})

        chain.reset_memory()
        print()

    # ── Rapport final ────────────────────────────────────────────────────────
    ok      = sum(1 for r in results if r["status"] == "✅ OK")
    blocked = sum(1 for r in results if r["blocked"])
    court   = sum(1 for r in results if r["short"])
    errors  = sum(1 for r in results if "ERREUR" in r["status"])
    avg_lat = total_time / len(results) if results else 0

    print(f"\n{'='*72}")
    print("  RAPPORT FINAL")
    print(f"{'='*72}")
    print(f"  ✅ OK           : {ok:2}/30  ({ok/30*100:.0f}%)")
    print(f"  ⛔ Bloquées     : {blocked:2}/30")
    print(f"  ⚠️  Trop courtes : {court:2}/30")
    print(f"  ❌ Erreurs      : {errors:2}/30")
    print(f"  ⏱  Latence moy  : {avg_lat:.0f}s")
    print(f"  ⏱  Temps total  : {total_time/60:.1f} min")
    print()

    # Résumé par domaine
    from collections import defaultdict
    by_domain: dict = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r["status"] == "✅ OK")
    print("  Par domaine :")
    for domain, statuses in sorted(by_domain.items()):
        ok_d = sum(statuses)
        total_d = len(statuses)
        bar = "✅" * ok_d + "⛔" * (total_d - ok_d)
        print(f"    {domain:<25} {bar}  {ok_d}/{total_d}")

    print()
    print("  Détail :")
    for r in results:
        src = f"({r['sources']} src)" if not r["blocked"] else ""
        print(f"    [{r['qid']}] {r['status']:<15} {r['domain']:<25} {r['latency']:.0f}s {src}")

    # Sauvegarde JSON
    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval30_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "total": 30, "ok": ok, "blocked": blocked, "short": court,
            "avg_latency_s": round(avg_lat, 1),
            "results": [{k: v for k, v in r.items() if k != "sources_list"} for r in results]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  Rapport sauvegardé : {out_file}")
    print(f"{'='*72}\n")

    return ok >= 21  # objectif 70%


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
