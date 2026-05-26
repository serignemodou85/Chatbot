"""Quick CLI test for the full RAG pipeline."""
import sys
import time

# Force UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retrieval.rag_chain import RAGChain

print("=== Initialisation RAGChain ===")
chain = RAGChain()
print()

tests = [
    ("C'est quoi Wazuh ?", "\U0001f393 Etudiant"),
    ("Comment installer l'agent Wazuh sur Ubuntu ?", "\U0001f5a5 Admin systeme"),
    ("Quelles sont les limites de Wazuh pour la detection d'intrusions ?", "\U0001f512 Pro cybersecurite"),
]

for question, mode in tests:
    label = mode.encode("ascii", "replace").decode()
    print(f"=== [{label}] ===")
    print(f"Q: {question}")
    t0 = time.time()
    result = chain.ask(question, mode)
    latency = time.time() - t0
    print(f"Latence : {latency:.1f}s")
    answer = result["answer"]
    # Safe print for Windows console
    safe_answer = answer.encode("cp1252", "replace").decode("cp1252")
    print(f"Reponse ({len(answer)} chars) :")
    print(safe_answer[:500])
    if len(answer) > 500:
        print("  [...]")
    print("Sources :")
    for s in result["sources"]:
        src = s['file'].encode("cp1252", "replace").decode("cp1252")
        print(f"  - {src} (p.{s['page']})")
    print()
