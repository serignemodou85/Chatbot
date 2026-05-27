# ── src/interface/app.py ──────────────────────────────────────────────────────
# Interface Streamlit du chatbot RAG.
#
# Lancement :
#   streamlit run src/interface/app.py
#
# Fonctionnalités :
#   - Historique multi-conversations dans la sidebar (comme ChatGPT)
#   - Mémoire LLM restaurée quand on reprend une conversation
#   - Chat avec sources documentaires + feedback 👍/👎
#   - Onglet Agents Phase 2 (CrewAI multi-agents)
#   - Sélection du mode utilisateur (étudiant / admin / pro)

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from loguru import logger

from src.retrieval.rag_chain import RAGChain
from src.agents.crew import CyberSecCrew

# ── Chemins ───────────────────────────────────────────────────────────────────
SESSION_DIR  = Path("data/session")
INDEX_FILE   = SESSION_DIR / "index.json"
FEEDBACK_FILE = Path("data/feedback/user_feedback.json")

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "Bonjour ! Je suis votre assistant cybersécurité.\n\n"
        "Je peux vous aider sur :\n"
        "- Configuration de pare-feux et VPN\n"
        "- Supervision avec Zabbix et Wazuh\n"
        "- Administration Linux / Windows Server\n"
        "- Analyse de logs et détection d'incidents\n\n"
        "Quelle est votre question ?"
    ),
    "sources": [],
}


# ── Gestion de l'index des conversations ─────────────────────────────────────

def _load_index() -> list:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        return []
    try:
        with INDEX_FILE.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        return sorted(entries, key=lambda x: x.get("updated_at", ""), reverse=True)
    except Exception:
        return []


def _save_index(entries: list):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _update_index_entry(conv_id: str, **kwargs):
    entries = _load_index()
    for entry in entries:
        if entry["id"] == conv_id:
            entry.update(kwargs)
            entry["updated_at"] = datetime.now(timezone.utc).isoformat()
            break
    _save_index(entries)


# ── Gestion des fichiers de conversation ──────────────────────────────────────

def _new_conv_id() -> str:
    return "conv_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _create_conversation() -> str:
    conv_id = _new_conv_id()
    now = datetime.now(timezone.utc).isoformat()
    entries = _load_index()
    entries.append({
        "id": conv_id,
        "title": "Nouvelle conversation",
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    })
    _save_index(entries)
    _save_conversation(conv_id, [dict(WELCOME_MESSAGE)])
    return conv_id


def _load_conversation(conv_id: str) -> list:
    path = SESSION_DIR / f"{conv_id}.json"
    if not path.exists():
        return [dict(WELCOME_MESSAGE)]
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return [dict(WELCOME_MESSAGE)]


def _save_conversation(conv_id: str, messages: list):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSION_DIR / f"{conv_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# ── Feedback ──────────────────────────────────────────────────────────────────

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


# ── Chargement du pipeline RAG (une seule fois par session) ───────────────────
@st.cache_resource(show_spinner="Chargement du pipeline RAG...")
def load_rag_chain():
    return RAGChain()


# ── Initialisation de la session ──────────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = load_rag_chain()

if "active_conv_id" not in st.session_state:
    index = _load_index()
    if index:
        latest = index[0]
        st.session_state.active_conv_id = latest["id"]
        st.session_state.messages = _load_conversation(latest["id"])
        st.session_state.rag_chain.restore_memory(st.session_state.messages)
    else:
        conv_id = _create_conversation()
        st.session_state.active_conv_id = conv_id
        st.session_state.messages = [dict(WELCOME_MESSAGE)]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Bouton nouvelle conversation ──────────────────────────────────────────
    if st.button("➕ Nouvelle conversation", use_container_width=True, type="primary"):
        conv_id = _create_conversation()
        st.session_state.active_conv_id = conv_id
        st.session_state.messages = [dict(WELCOME_MESSAGE)]
        st.session_state.rag_chain.reset_memory()
        st.rerun()

    st.divider()

    # ── Liste des conversations ───────────────────────────────────────────────
    st.caption("💬 Conversations")
    index = _load_index()

    if not index:
        st.caption("_Aucune conversation_")
    else:
        for entry in index:
            conv_id = entry["id"]
            title   = entry.get("title", "Nouvelle conversation")
            is_active = (conv_id == st.session_state.active_conv_id)

            # Tronquer à 36 caractères pour la sidebar
            short = (title[:36] + "…") if len(title) > 36 else title
            label = ("▶ " + short) if is_active else short

            if st.button(label, key=f"conv_{conv_id}", use_container_width=True,
                         disabled=is_active):
                msgs = _load_conversation(conv_id)
                st.session_state.active_conv_id = conv_id
                st.session_state.messages = msgs
                st.session_state.rag_chain.restore_memory(msgs)
                st.rerun()

    st.divider()

    # ── Mode utilisateur ──────────────────────────────────────────────────────
    user_mode = st.selectbox(
        "Mode utilisateur",
        ["🎓 Étudiant", "🖥️ Admin système", "🔒 Pro cybersécurité"],
        help="Adapte le niveau de détail des réponses",
        key="user_mode",
    )

    st.divider()
    st.caption("📚 Base documentaire")
    st.caption("Wazuh · Zabbix · VPN · Firewall · Linux · Windows")
    st.divider()
    st.caption("Phase 1 — RAG Chat · Phase 2 — Agents")
    st.caption("LangChain + ChromaDB + CrewAI + Llama3")


# ── Corps principal ───────────────────────────────────────────────────────────
st.title("🛡️ CyberSec RAG Chatbot")
st.caption("Assistant IA spécialisé en cybersécurité, réseau et administration système")

tab1, tab2 = st.tabs(["💬 Chat RAG", "🤖 Agents — Phase 2"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat RAG (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Historique de la conversation active ──────────────────────────────────
    for msg_index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                        "Feedback enregistré : 👍" if feedback_value == "up"
                        else "Feedback enregistré : 👎"
                    )
                else:
                    col_up, col_down = st.columns(2)
                    with col_up:
                        if st.button("👍 Utile", key=f"up_{msg_index}", use_container_width=True):
                            entry = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "mode_utilisateur": st.session_state.get("user_mode", user_mode),
                                "feedback": "up",
                                "conv_id": st.session_state.active_conv_id,
                                "question": _find_previous_user_message(st.session_state.messages, msg_index),
                                "answer": msg.get("content", ""),
                                "sources": msg.get("sources", []),
                            }
                            _append_feedback(entry)
                            st.session_state.messages[msg_index]["feedback"] = "up"
                            _save_conversation(st.session_state.active_conv_id, st.session_state.messages)
                            st.rerun()
                    with col_down:
                        if st.button("👎 À améliorer", key=f"down_{msg_index}", use_container_width=True):
                            entry = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "mode_utilisateur": st.session_state.get("user_mode", user_mode),
                                "feedback": "down",
                                "conv_id": st.session_state.active_conv_id,
                                "question": _find_previous_user_message(st.session_state.messages, msg_index),
                                "answer": msg.get("content", ""),
                                "sources": msg.get("sources", []),
                            }
                            _append_feedback(entry)
                            st.session_state.messages[msg_index]["feedback"] = "down"
                            _save_conversation(st.session_state.active_conv_id, st.session_state.messages)
                            st.rerun()

    # ── Zone de saisie ────────────────────────────────────────────────────────
    if question := st.chat_input("Posez votre question sur la cybersécurité..."):

        st.session_state.messages.append({"role": "user", "content": question, "sources": []})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans la base documentaire..."):
                try:
                    result = st.session_state.rag_chain.ask(
                        question,
                        user_mode=st.session_state.get("user_mode", user_mode),
                    )
                    answer  = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer  = f"⚠️ Erreur lors de la génération : {e}"
                    sources = []
                    logger.error(f"Erreur RAG : {e}")

            st.markdown(answer)

            if sources:
                with st.expander(f"📄 Sources utilisées ({len(sources)})", expanded=True):
                    for src in sources:
                        file_name = Path(src["file"]).name
                        page_info = f" — page {src['page']}" if src.get("page") else ""
                        st.caption(f"📎 **{file_name}**{page_info}")
                        st.caption(f"_{src['excerpt']}_")
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        _save_conversation(st.session_state.active_conv_id, st.session_state.messages)

        # Titre = première question de l'utilisateur (une seule fois)
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        title_kwargs = {"message_count": len(st.session_state.messages)}
        if len(user_msgs) == 1:
            title = (question[:45] + "…") if len(question) > 45 else question
            title_kwargs["title"] = title
        _update_index_entry(st.session_state.active_conv_id, **title_kwargs)


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

        from src.agents.crew import _classify
        domains = _classify(agent_question)
        domain_labels = {"doc": "🗂️ Documentation", "network": "🌐 Réseau", "security": "🔐 Sécurité"}
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

    if "last_crew_result" in st.session_state:
        result = st.session_state["last_crew_result"]

        st.divider()
        st.subheader("📋 Rapport final")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mode", result.get("user_mode", "—"))
        with col2:
            st.metric("Agents utilisés", len(result.get("agents_used", [])))
        with col3:
            st.metric("Domaines", " · ".join(result.get("domains", [])))

        st.markdown(result.get("report", "Aucun rapport généré."))

        if st.button("🗑️ Effacer le rapport", key="clear_report"):
            st.session_state.pop("last_crew_result", None)
            st.rerun()
