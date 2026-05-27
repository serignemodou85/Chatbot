# ── src/interface/app.py ──────────────────────────────────────────────────────
# Interface web du chatbot RAG, construite avec Streamlit.
#
# Lancement :
#   streamlit run src/interface/app.py
#
# Fonctionnalités :
#   - Chat avec historique visuel
#   - Affichage des sources documentaires
#   - Bouton de reset de la conversation
#   - Sélection du mode utilisateur (étudiant / admin / pro)
#   - Onglet Agents Phase 2 (CrewAI multi-agents)

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Ajouter la racine du projet au path Python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from loguru import logger

from src.retrieval.rag_chain import RAGChain
from src.agents.crew import CyberSecCrew

FEEDBACK_FILE = Path("data/feedback/user_feedback.json")
SESSION_FILE = Path("data/session/chat_history.json")


def _save_session(messages: list):
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_FILE.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def _load_session() -> list:
    if not SESSION_FILE.exists():
        return []
    try:
        with SESSION_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _load_feedback_entries():
    if not FEEDBACK_FILE.exists():
        return []
    try:
        with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _append_feedback(entry: dict):
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    entries = _load_feedback_entries()
    entries.append(entry)
    with FEEDBACK_FILE.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _find_previous_user_message(messages, current_index: int) -> str:
    for idx in range(current_index - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return messages[idx].get("content", "")
    return ""


# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberSec RAG Chatbot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Initialisation du pipeline RAG (une seule fois par session) ───────────────
@st.cache_resource(show_spinner="Chargement du pipeline RAG...")
def load_rag_chain():
    """
    @cache_resource : Streamlit ne recrée l'objet qu'une seule fois
    même si l'utilisateur recharge la page. Évite de recharger
    ChromaDB et le modèle d'embeddings à chaque interaction.
    """
    return RAGChain()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(" Paramètres")

    # Mode utilisateur : influence le prompt (Phase 2)
    user_mode = st.selectbox(
        "Mode utilisateur",
        ["🎓 Étudiant", "🖥️ Admin système", "🔒 Pro cybersécurité"],
        help="Adapte le niveau de détail des réponses",
        key="user_mode",
    )

    st.divider()

    # Bouton de reset de la conversation
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        SESSION_FILE.unlink(missing_ok=True)
        try:
            st.session_state.rag_chain.reset_memory()
        except Exception:
            pass
        st.rerun()

    st.divider()

    # Informations sur la base documentaire
    st.caption("📚 Base documentaire")
    st.caption("Wazuh · Zabbix · VPN · Firewall · Linux · Windows")

    st.divider()
    st.caption("Phase 1 — RAG Chat · Phase 2 — Agents")
    st.caption("LangChain + ChromaDB + CrewAI + Llama3")


# ── Corps principal ───────────────────────────────────────────────────────────
st.title("🛡️ CyberSec RAG Chatbot")
st.caption("Assistant IA spécialisé en cybersécurité, réseau et administration système")

# Initialiser l'historique — récupérer depuis le fichier si dispo (survie à la veille)
if "messages" not in st.session_state:
    saved = _load_session()
    if saved:
        st.session_state.messages = saved
    else:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis votre assistant cybersécurité.\n\n"
                "Je peux vous aider sur :\n"
                "-  Configuration de pare-feux et VPN\n"
                "-  Supervision avec Zabbix et Wazuh\n"
                "-  Administration Linux / Windows Server\n"
                "-  Analyse de logs et détection d'incidents\n\n"
                "Quelle est votre question ?"
            ),
            "sources": [],
        }]

# Stocker la chain dans la session pour pouvoir reset la mémoire
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = load_rag_chain()

# ── Onglets Phase 1 / Phase 2 ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat RAG", "🤖 Agents — Phase 2"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat RAG (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Affichage de l'historique ─────────────────────────────────────────────
    for msg_index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Afficher les sources si disponibles
            if msg.get("sources"):
                with st.expander(f"📄 Sources ({len(msg['sources'])})", expanded=False):
                    for src in msg["sources"]:
                        file_name = Path(src["file"]).name
                        page_info = f" — page {src['page']}" if src.get("page") else ""
                        st.caption(f"📎 **{file_name}**{page_info}")
                        st.caption(f"_{src['excerpt']}_")
                        st.divider()

            if msg["role"] == "assistant":
                feedback_value = msg.get("feedback")
                if feedback_value:
                    st.caption(
                        "Feedback enregistré : 👍"
                        if feedback_value == "up"
                        else "Feedback enregistré : 👎"
                    )
                else:
                    col_up, col_down = st.columns(2)
                    with col_up:
                        if st.button("👍 Utile", key=f"feedback_up_{msg_index}", use_container_width=True):
                            feedback_entry = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "mode_utilisateur": st.session_state.get("user_mode", user_mode),
                                "feedback": "up",
                                "question": _find_previous_user_message(st.session_state.messages, msg_index),
                                "answer": msg.get("content", ""),
                                "sources": msg.get("sources", []),
                                "message_index": msg_index,
                            }
                            _append_feedback(feedback_entry)
                            st.session_state.messages[msg_index]["feedback"] = "up"
                            st.rerun()
                    with col_down:
                        if st.button("👎 À améliorer", key=f"feedback_down_{msg_index}", use_container_width=True):
                            feedback_entry = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "mode_utilisateur": st.session_state.get("user_mode", user_mode),
                                "feedback": "down",
                                "question": _find_previous_user_message(st.session_state.messages, msg_index),
                                "answer": msg.get("content", ""),
                                "sources": msg.get("sources", []),
                                "message_index": msg_index,
                            }
                            _append_feedback(feedback_entry)
                            st.session_state.messages[msg_index]["feedback"] = "down"
                            st.rerun()

    # ── Zone de saisie ────────────────────────────────────────────────────────
    if question := st.chat_input("Posez votre question sur la cybersécurité..."):

        # Ajouter la question de l'utilisateur à l'historique
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "sources": [],
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans la base documentaire..."):
                try:
                    result = st.session_state.rag_chain.ask(
                        question,
                        user_mode=st.session_state.get("user_mode", user_mode),
                    )
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"⚠️ Erreur lors de la génération : {e}"
                    sources = []
                    logger.error(f"Erreur RAG : {e}")

            st.markdown(answer)

            # Afficher les sources de la réponse courante
            if sources:
                with st.expander(f"📄 Sources utilisées ({len(sources)})", expanded=True):
                    for src in sources:
                        file_name = Path(src["file"]).name
                        page_info = f" — page {src['page']}" if src.get("page") else ""
                        st.caption(f"📎 **{file_name}**{page_info}")
                        st.caption(f"_{src['excerpt']}_")
                        st.divider()

        # Sauvegarder la réponse dans l'historique + persistance sur disque
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        _save_session(st.session_state.messages)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Agents Phase 2 (CrewAI)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Analyse multi-agents")
    st.caption(
        "Posez une question complexe : plusieurs agents spécialisés collaborent "
        "pour produire un rapport structuré."
    )

    st.info(
        "**Agents disponibles :**  \n"
        "🗂️ **Documentation** — recherche générale dans la base  \n"
        "🌐 **Réseau** — VPN, pare-feu, pfSense, VLAN, routage  \n"
        "🔐 **Sécurité** — Wazuh, Zabbix, SIEM, audits, CVE  \n"
        "📝 **Rapport** — synthèse finale adaptée à votre profil",
        icon="ℹ️",
    )

    agent_question = st.text_area(
        "Question complexe pour les agents",
        placeholder=(
            "Ex : Compare Wazuh et Zabbix pour superviser un réseau de 50 machines.\n"
            "Ex : Comment configurer un VPN site-à-site avec pfSense ?\n"
            "Ex : Quels sont les risques d'un serveur Ubuntu sans pare-feu ?"
        ),
        height=120,
        key="agent_question",
    )

    launch_btn = st.button(
        "🚀 Lancer l'analyse",
        type="primary",
        use_container_width=True,
        disabled=not agent_question.strip(),
    )

    if launch_btn and agent_question.strip():
        current_mode = st.session_state.get("user_mode", user_mode)

        # Déterminer les agents qui seront activés (aperçu avant lancement)
        from src.agents.crew import _classify
        domains = _classify(agent_question)
        domain_labels = {
            "doc": "🗂️ Documentation",
            "network": "🌐 Réseau",
            "security": "🔐 Sécurité",
        }
        active_labels = [domain_labels.get(d, d) for d in domains]
        st.caption(f"Agents activés : {' · '.join(active_labels)} → 📝 Rapport")

        with st.spinner("Analyse en cours... (peut prendre 1-2 minutes)"):
            try:
                crew = CyberSecCrew()
                result = crew.run(agent_question, current_mode)
                st.session_state["last_crew_result"] = result
            except Exception as e:
                st.error(f"⚠️ Erreur lors de l'analyse : {e}")
                logger.error(f"Erreur Crew : {e}")
                st.session_state.pop("last_crew_result", None)

    # Afficher le dernier rapport si disponible
    if "last_crew_result" in st.session_state:
        result = st.session_state["last_crew_result"]

        st.divider()
        st.subheader("📋 Rapport final")

        # Méta-infos
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.metric("Mode", result.get("user_mode", "—"))
        with meta_col2:
            agents_used = result.get("agents_used", [])
            st.metric("Agents utilisés", len(agents_used))
        with meta_col3:
            domains_used = result.get("domains", [])
            st.metric("Domaines", " · ".join(domains_used))

        st.markdown(result.get("report", "Aucun rapport généré."))

        # Bouton pour effacer le rapport
        if st.button("🗑️ Effacer le rapport", key="clear_report"):
            st.session_state.pop("last_crew_result", None)
            st.rerun()
