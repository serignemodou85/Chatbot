# ── src/agents/tasks.py ──────────────────────────────────────────────────────
# Fabrique de tâches CrewAI pour les agents Phase 2.

from crewai import Agent, Task

_MODE_INSTRUCTIONS = {
    "\U0001f393 Étudiant": (
        "Langage accessible, définir les termes techniques, utiliser des analogies. "
        "Structurer en étapes simples et progressives."
    ),
    "\U0001f5a5️ Admin système": (
        "Procédures concrètes et opérationnelles, commandes exactes, "
        "points de vérification. Format checklist si applicable."
    ),
    "\U0001f512 Pro cybersécurité": (
        "Analyse technique approfondie, évaluation des risques, "
        "alternatives défensives, références aux standards (NIST, MITRE ATT&CK)."
    ),
}


def make_research_task(agent: Agent, question: str, domain: str) -> Task:
    return Task(
        description=(
            f"Recherche les informations pertinentes pour répondre à cette question :\n"
            f"'{question}'\n\n"
            f"Domaine de focus : {domain}\n\n"
            f"Instructions :\n"
            f"1. Lance une recherche avec la question originale\n"
            f"2. Si les résultats manquent de précision, reformule et relance\n"
            f"3. Extrais : procédures, commandes, configurations, concepts clés\n"
            f"4. Note les sources (nom du fichier, page si disponible)\n"
            f"5. Retourne un résumé structuré de tes trouvailles"
        ),
        expected_output=(
            "Un résumé structuré en français contenant :\n"
            "- Les informations clés trouvées dans les documents\n"
            "- Les commandes ou configurations si applicable\n"
            "- Les sources consultées (nom fichier + page)\n"
            "- Ce qui n'a pas été trouvé si pertinent"
        ),
        agent=agent,
    )


def make_report_task(
    agent: Agent,
    question: str,
    user_mode: str,
    context: str,
) -> Task:
    mode_instruction = _MODE_INSTRUCTIONS.get(
        user_mode,
        "Équilibre pédagogie et technicité, adapté à un public IT généraliste.",
    )

    return Task(
        description=(
            f"Produis le rapport final en réponse à la question :\n"
            f"'{question}'\n\n"
            f"Mode utilisateur : {user_mode}\n"
            f"Instruction de style : {mode_instruction}\n\n"
            f"Analyses reçues des agents spécialistes :\n"
            f"{context}\n\n"
            f"Structure obligatoire du rapport :\n"
            f"1. **Résumé** (2-3 phrases, réponse directe)\n"
            f"2. **Analyse détaillée** (adaptée au mode utilisateur)\n"
            f"3. **Commandes / Configuration** (si applicable, en blocs de code)\n"
            f"4. **Points clés à retenir**\n"
            f"5. **Sources consultées**"
        ),
        expected_output=(
            "Un rapport complet en Markdown, en français, "
            "structuré selon les 5 sections demandées, "
            "directement exploitable par l'utilisateur."
        ),
        agent=agent,
    )
