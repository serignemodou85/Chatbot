# ── src/retrieval/rag_chain.py ────────────────────────────────────────────────
# Cœur du système RAG.
# Assemble : Retriever + PromptTemplate + LLM + Mémoire de conversation.
#
# Utilisation :
#   chain = RAGChain()
#   result = chain.ask("Comment configurer Wazuh ?")
#   print(result["answer"])
#   print(result["sources"])

from typing import Any, Dict, List

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from loguru import logger

from config.settings import settings
from src.llm.llm_factory import get_llm
from src.ingestion.vectorstore import VectorStoreManager


# ── Prompt système ────────────────────────────────────────────────────────────
# Ce prompt est injecté à chaque appel. Il définit le comportement du LLM.
# {context} = chunks récupérés par ChromaDB
# {question} = question de l'utilisateur
# {chat_history} = historique géré automatiquement par ConversationBufferWindowMemory

RAG_PROMPT_TEMPLATE = """Tu es un assistant expert en cybersécurité, \
administration système et infrastructure réseau.
Tu réponds en français et en Anglais , de manière précise et structurée.

Adapte ton niveau de détail selon le contexte :
- Question débutant → explication claire avec analogies
- Question technique → commandes, configuration, exemples concrets
- Question d'analyse → interprétation approfondie

Utilise UNIQUEMENT les informations du contexte ci-dessous pour répondre.
Si la réponse n'est pas dans le contexte, dis-le clairement plutôt que d'inventer.

─────────────────────────────────────────────
CONTEXTE DOCUMENTAIRE :
{context}
─────────────────────────────────────────────
HISTORIQUE DE CONVERSATION :
{chat_history}
─────────────────────────────────────────────

QUESTION : {question}

RÉPONSE :"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


class RAGChain:
    """
    Pipeline RAG conversationnel complet.

    Fonctionnement à chaque appel à ask() :
    1. La question est vectorisée (embedding)
    2. ChromaDB retourne les K chunks les plus proches sémantiquement
    3. Les chunks + l'historique + la question sont injectés dans le prompt
    4. Le LLM génère une réponse basée uniquement sur le contexte
    5. La réponse + la question sont ajoutées à la mémoire
    """

    def __init__(self):
        logger.info("Initialisation du pipeline RAG...")

        # Chargement de la base vectorielle
        self.vs_manager = VectorStoreManager()
        retriever = self.vs_manager.get_retriever()

        # Mémoire glissante : garde les N derniers échanges
        # window_k=5 = 5 tours de conversation conservés
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=5,
        )

        # LLM (OpenAI ou Ollama selon .env)
        llm = get_llm()

        # Assemblage de la chain LangChain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True,  # Inclure les sources dans la réponse
            verbose=False,
        )

        logger.info("Pipeline RAG prêt.")

    def ask(self, question: str, user_mode: str = "🖥️ Admin système") -> Dict[str, Any]:
        """
        Pose une question au chatbot.

        Retourne un dict avec :
          - answer  : la réponse générée
          - sources : liste des documents sources utilisés
          - question: la question posée (pour l'affichage)
        """
        if not question.strip():
            return {"answer": "Merci de poser une question.", "sources": []}

        logger.info(f"Question : {question[:80]}...")

        mode_instruction = self._build_mode_instruction(user_mode)
        question_with_mode = (
            f"[Mode utilisateur] {mode_instruction}\n"
            f"[Question]\n{question}"
        )
        result = self.chain.invoke({"question": question_with_mode})

        # Extraction et déduplication des sources
        sources = self._extract_sources(result.get("source_documents", []))

        return {
            "question": question,
            "answer": result["answer"],
            "sources": sources,
        }

    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
        """
        Formate les métadonnées des documents sources.
        Déduplique par nom de fichier pour éviter les répétitions.
        """
        seen = set()
        sources = []

        for doc in docs:
            meta = doc.metadata
            source = meta.get("source", "Source inconnue")
            page = meta.get("page", None)

            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": source,
                    "page": page,
                    "excerpt": doc.page_content[:200] + "...",
                })

        return sources

    def reset_memory(self):
        """Efface l'historique de conversation (nouveau sujet)."""
        self.memory.clear()
        logger.info("Mémoire de conversation effacée.")

    def restore_memory(self, messages: list, k: int = 5):
        """
        Reconstruit la mémoire depuis un historique JSON sauvegardé.
        Rejoue les k dernières paires user/assistant dans l'objet mémoire.
        Appelé quand on recharge une conversation existante.
        """
        self.memory.clear()
        pairs = []
        pending_user = None
        for msg in messages:
            if msg.get("role") == "user":
                pending_user = msg.get("content", "")
            elif msg.get("role") == "assistant" and pending_user is not None:
                pairs.append((pending_user, msg.get("content", "")))
                pending_user = None

        for user_text, assistant_text in pairs[-k:]:
            self.memory.chat_memory.add_user_message(user_text)
            self.memory.chat_memory.add_ai_message(assistant_text)

        if pairs:
            logger.info(f"Mémoire restaurée : {min(len(pairs), k)} échange(s) rechargé(s)")

    @staticmethod
    def _build_mode_instruction(user_mode: str) -> str:
        """Construit une consigne de style selon le mode sélectionné dans l'UI."""
        mode_map = {
            "🎓 Étudiant": (
                "Public débutant/intermédiaire. Utilise un langage simple, "
                "définit les termes techniques et propose des étapes pédagogiques."
            ),
            "🖥️ Admin système": (
                "Public opérationnel IT. Donne des procédures concrètes, des commandes "
                "et des points de vérification exploitables en production."
            ),
            "🔒 Pro cybersécurité": (
                "Public expert sécurité. Réponds de façon concise et technique, avec "
                "analyse de risques, limites, et alternatives défensives."
            ),
        }
        return mode_map.get(
            user_mode,
            "Public opérationnel IT. Répondre avec équilibre entre pédagogie et technicité.",
        )
