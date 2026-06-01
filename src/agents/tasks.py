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
        "alternatives défensives. Cite uniquement les sources présentes dans "
        "le contexte fourni — ne jamais inventer de références externes."
    ),
}


def make_research_task(agent: Agent, question: str, domain: str) -> Task:
    return Task(
        description=(
            f"Recherche les informations pertinentes pour répondre à cette question :\n"
            f"'{question}'\n\n"
            f"Domaine de focus : {domain}\n\n"
            f"PROCÉDURE OBLIGATOIRE (une seule étape) :\n"
            f"1. Utilise l'outil de recherche UNE SEULE fois\n"
            f"2. Dès que tu reçois les documents, écris IMMÉDIATEMENT :\n"
            f"   Final Answer: [ton résumé structuré]\n"
            f"   NE fais PAS d'autre Action après avoir reçu les résultats."
        ),
        expected_output=(
            "Un résumé structuré en français contenant :\n"
            "- Les informations clés trouvées dans les documents\n"
            "- Les commandes ou configurations exactes telles qu'elles apparaissent dans les docs\n"
            "- Les sources (nom fichier + page retournés par l'outil de recherche)\n"
            "- Ce qui n'a pas été trouvé si pertinent\n"
            "IMPORTANT : une seule recherche suffit. Ne pas inventer de commandes ni d'URLs."
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
            f"⚠️ RÈGLE N°1 — AVANT D'ÉCRIRE QUOI QUE CE SOIT :\n"
            f"Chaque commande, nom de fichier ou procédure que tu écris DOIT exister "
            f"MOT POUR MOT dans le CONTEXTE DOCUMENTAIRE ci-dessous. "
            f"Si tu ne trouves pas une commande dans le contexte → supprime la section 'Commandes'. "
            f"N'utilise JAMAIS tes connaissances d'entraînement. "
            f"Si le contexte porte sur un sujet différent de la question "
            f"(ex: question SSH mais contexte IPsec) → indique-le clairement "
            f"et réponds UNIQUEMENT avec ce que le contexte contient.\n\n"
            f"QUESTION : '{question}'\n\n"
            f"MODE UTILISATEUR : {user_mode} — STYLE : {mode_instruction}\n\n"
            f"────────────────────────────────────────────\n"
            f"CONTEXTE DOCUMENTAIRE (extraits réels de la base) :\n"
            f"{context}\n"
            f"────────────────────────────────────────────\n\n"
            f"RÈGLES STRICTES :\n"
            f"1. SOURCE UNIQUE — Uniquement le contexte ci-dessus. Jamais tes connaissances d'entraînement.\n"
            f"2. COMMANDES — Section présente SEULEMENT si la commande existe mot pour mot "
            f"dans le contexte. Sinon, omettre complètement la section. "
            f"Ne jamais adapter, combiner, ni inventer.\n"
            f"3. FUTUR INTERDIT — Rédige au présent. Pas de 'il faudra', 'vous devrez', 'je vais'.\n"
            f"4. INSUFFISANCE — Si le contexte ne répond pas à la question : "
            f"'Le contexte documentaire ne contient pas d'information suffisante sur ce point.'\n"
            f"5. SOURCES — Uniquement les fichiers et pages du contexte. Jamais d'URL inventée.\n\n"
            f"Structure du rapport :\n"
            f"1. **Résumé** (2-3 phrases, réponse directe)\n"
            f"2. **Analyse détaillée** (adaptée au mode)\n"
            f"3. **Commandes / Configuration** (SEULEMENT si présentes mot pour mot dans le contexte)\n"
            f"4. **Points clés**\n"
            f"5. **Sources** (fichiers + pages du contexte uniquement)"
        ),
        expected_output=(
            "Un rapport Markdown en français, basé strictement sur le contexte fourni. "
            "Aucune formulation au futur. Aucune commande inventée. Aucune URL externe. "
            "Commandes = uniquement celles présentes mot pour mot dans le contexte. "
            "Sources = uniquement les chemins de fichiers du contexte."
        ),
        agent=agent,
    )
