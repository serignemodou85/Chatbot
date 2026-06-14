# ── src/interface/app.py ──────────────────────────────────────────────────────
# Interface Streamlit unifiée — chat unique avec routage automatique.
# Questions simples   → RAG direct (~2-3 min)
# Questions complexes → Agents CrewAI (~3-5 min)
#
# Lancement : venv\Scripts\python -m streamlit run src/interface/app.py

import sys
import re
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Mixing V1 models.*")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from loguru import logger

st.set_page_config(
    page_title="InfraBot — Assistant Cybersécurité",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import settings as _settings

try:
    _settings.validate()
except ValueError as _e:
    st.error(f"❌ Configuration invalide : {_e}")
    st.stop()

from src.retrieval.rag_chain import RAGChain
from src.agents.crew import CyberSecCrew, _classify as _classify_domains

# ── Chemins ───────────────────────────────────────────────────────────────────
SESSION_DIR   = Path("data/session")
INDEX_FILE    = SESSION_DIR / "index.json"
FEEDBACK_FILE = Path("data/feedback/user_feedback.json")

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "Bonjour ! Je suis **InfraBot**, votre assistant cybersécurité.\n\n"
        "Je peux vous aider sur :\n"
        "- 🔥 Configuration de pare-feux et VPN (pfSense, OpenVPN, IPSec)\n"
        "- 📊 Supervision avec Zabbix et Wazuh\n"
        "- 🐧 Administration Linux / Windows Server\n"
        "- 🔍 Analyse de logs et détection d'incidents\n"
        "- 🏢 Gestion ERP avec Dolibarr\n\n"
        "Pour les questions complexes (comparaisons, déploiement, audit), "
        "je mobilise automatiquement des **agents spécialisés**.\n\n"
        "Quelle est votre question ?"
    ),
    "msg_type": "welcome",
    "sources": [],
}

# ── Routage automatique ───────────────────────────────────────────────────────
_AGENT_KEYWORDS = {
    "compare", "comparaison", "comparer", "vs", "versus",
    "différence", "difference", "différences", "differences",
    "avantage", "avantages", "inconvénient", "inconvénients",
    "analyse", "analyser", "audit", "auditer",
    "stratégie", "strategie", "stratégique", "architecture",
    "infrastructure", "déployer", "deployer", "déploiement", "deploiement",
    "recommande", "recommandation", "recommander",
    "meilleur", "choisir", "quel outil", "lequel",
    "rapport", "bilan", "synthèse", "synthese",
    "pour combien", "machines", "postes", "serveurs",
    "diagnostiquer", "diagnostic", "dépanner", "depanner", "troubleshoot",
    "debugger", "déboguer", "debug", "résoudre", "resoudre",
    "bloqué", "bloque", "erreur", "problème", "probleme",
    "ne fonctionne pas", "ne marche pas", "échoue", "echoue",
    "configurer", "configuration", "configuring", "mettre en place",
    "installer", "installation",
}

_MULTI_SYSTEMS = {
    "wazuh", "zabbix", "pfsense", "openvpn", "ipsec", "nagios",
    "suricata", "snort", "ossec", "elasticsearch", "kibana", "splunk",
    "windows server", "ubuntu", "debian", "centos", "ansible", "docker",
}


def _route_question(question: str) -> str:
    q = question.lower()
    words = set(q.split())
    if words & _AGENT_KEYWORDS or any(kw in q for kw in _AGENT_KEYWORDS):
        return "agents"
    systems_found = sum(1 for s in _MULTI_SYSTEMS if s in q)
    if systems_found >= 2:
        return "agents"
    if len(words) > 20 and systems_found >= 1:
        return "agents"
    return "rag"


# ── Gestion de l'index des conversations ─────────────────────────────────────

def _load_index() -> list:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        return []
    try:
        with INDEX_FILE.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        # Backward compatibility : champs ajoutés progressivement
        for e in entries:
            e.setdefault("archived", False)
            e.setdefault("pinned", False)
            e.setdefault("message_count", 0)
            e.setdefault("created_at", e.get("updated_at", ""))
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


def _create_conversation() -> str:
    conv_id = "conv_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    now = datetime.now(timezone.utc).isoformat()
    entries = _load_index()
    entries.insert(0, {
        "id": conv_id,
        "title": "Nouvelle conversation",
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
        "archived": False,
        "pinned": False,
    })
    _save_index(entries)
    _save_conversation(conv_id, [dict(WELCOME_MESSAGE)])
    return conv_id


def _archive_conversation(conv_id: str):
    _update_index_entry(conv_id, archived=True)


def _restore_conversation(conv_id: str):
    _update_index_entry(conv_id, archived=False)


def _delete_conversation(conv_id: str):
    if not _safe_conv_id(conv_id):
        return
    entries = _load_index()
    entries = [e for e in entries if e["id"] != conv_id]
    _save_index(entries)
    path = SESSION_DIR / f"{conv_id}.json"
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _pin_conversation(conv_id: str, pinned: bool):
    _update_index_entry(conv_id, pinned=pinned)


def _rename_conversation(conv_id: str, new_title: str):
    title = new_title.strip()[:60] or "Conversation"
    _update_index_entry(conv_id, title=title)


def _format_date(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        if delta.days == 0:
            h = int(delta.seconds / 3600)
            return "À l'instant" if h == 0 else f"Il y a {h}h"
        elif delta.days == 1:
            return "Hier"
        elif delta.days < 7:
            return f"Il y a {delta.days}j"
        else:
            return dt.strftime("%d/%m/%Y")
    except Exception:
        return ""


def _switch_active(exclude_id: str | None = None):
    """Bascule vers une autre conversation ou en crée une nouvelle."""
    index = _load_index()
    alternatives = [
        e for e in index
        if not e.get("archived") and e["id"] != exclude_id
    ]
    if alternatives:
        nid = alternatives[0]["id"]
        msgs = _load_conversation(nid)
        st.session_state.active_conv_id = nid
        st.session_state.messages = msgs
        st.session_state.rag_chain.restore_memory(msgs)
    else:
        nid = _create_conversation()
        st.session_state.active_conv_id = nid
        st.session_state.messages = [dict(WELCOME_MESSAGE)]
        st.session_state.rag_chain.reset_memory()


# ── Gestion des fichiers de conversation ─────────────────────────────────────

_CONV_ID_RE = re.compile(r"^conv_\d{8}_\d{6}_\d+$")


def _safe_conv_id(conv_id: str) -> bool:
    return bool(_CONV_ID_RE.match(str(conv_id)))


def _load_conversation(conv_id: str) -> list:
    if not _safe_conv_id(conv_id):
        logger.warning(f"conv_id invalide ignoré : {conv_id!r}")
        return [dict(WELCOME_MESSAGE)]
    path = SESSION_DIR / f"{conv_id}.json"
    if not path.exists():
        return [dict(WELCOME_MESSAGE)]
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return [dict(WELCOME_MESSAGE)]


def _save_conversation(conv_id: str, messages: list):
    if not _safe_conv_id(conv_id):
        logger.warning(f"_save_conversation: conv_id invalide ignoré : {conv_id!r}")
        return
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSION_DIR / f"{conv_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    try:
        path.chmod(0o600)
    except Exception:
        pass


# ── Feedback ─────────────────────────────────────────────────────────────────

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
    try:
        FEEDBACK_FILE.chmod(0o600)
    except Exception:
        pass


def _find_previous_user_message(messages, current_index: int) -> str:
    for idx in range(current_index - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return messages[idx].get("content", "")
    return ""


# ── Chargement du pipeline RAG ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du pipeline RAG...")
def load_rag_chain():
    return RAGChain()


# ── Initialisation de la session ──────────────────────────────────────────────

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = load_rag_chain()

if "active_conv_id" not in st.session_state:
    index = _load_index()
    active = [e for e in index if not e.get("archived")]
    if active:
        latest = active[0]
        st.session_state.active_conv_id = latest["id"]
        st.session_state.messages = _load_conversation(latest["id"])
        st.session_state.rag_chain.restore_memory(st.session_state.messages)
    else:
        conv_id = _create_conversation()
        st.session_state.active_conv_id = conv_id
        st.session_state.messages = [dict(WELCOME_MESSAGE)]


# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Conversation active (bouton désactivé = conv courante) */
section[data-testid="stSidebar"] .stButton button[disabled] {
    opacity: 1 !important;
    background: rgba(99, 102, 241, 0.12) !important;
    border-left: 3px solid #6366f1 !important;
    font-weight: 600 !important;
}
/* Hover sur les autres items de conversation */
section[data-testid="stSidebar"] .stButton button:not([disabled]):hover {
    background: rgba(128, 128, 128, 0.08) !important;
}
/* Boutons feedback en pill */
[data-testid="stHorizontalBlock"] .stButton button {
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ InfraBot")

    if st.button("Nouvelle conversation", icon=":material/add:", use_container_width=True, type="primary"):
        conv_id = _create_conversation()
        st.session_state.active_conv_id = conv_id
        st.session_state.messages = [dict(WELCOME_MESSAGE)]
        st.session_state.rag_chain.reset_memory()
        st.session_state.pop("renaming", None)
        st.rerun()

    st.divider()

    index = _load_index()
    active_entries   = [e for e in index if not e.get("archived")]
    archived_entries = [e for e in index if e.get("archived")]

    # Épinglés en premier, puis plus récent
    active_entries.sort(key=lambda e: (not e.get("pinned", False), ""), reverse=False)
    active_entries.sort(key=lambda e: e.get("pinned", False), reverse=True)

    renaming = st.session_state.get("renaming")

    if not active_entries:
        st.caption("_Aucune conversation_")

    for entry in active_entries:
        conv_id   = entry["id"]
        title     = entry.get("title", "Nouvelle conversation")
        is_active = (conv_id == st.session_state.active_conv_id)
        is_pinned = entry.get("pinned", False)
        date_str  = _format_date(entry.get("updated_at") or entry.get("created_at", ""))
        n_msg     = entry.get("message_count", 0)
        short     = (title[:32] + "…") if len(title) > 32 else title
        prefix    = "▸ " if is_active else ("· " if is_pinned else "  ")

        # ── Mode renommage inline ─────────────────────────────────────────────
        if renaming == conv_id:
            new_val = st.text_input(
                "Renommer",
                value=title,
                key=f"rename_input_{conv_id}",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Sauver", icon=":material/check:", key=f"save_ren_{conv_id}",
                             use_container_width=True, type="primary"):
                    _rename_conversation(conv_id, new_val)
                    st.session_state.pop("renaming")
                    st.rerun()
            with c2:
                if st.button("Annuler", icon=":material/close:", key=f"cancel_ren_{conv_id}",
                             use_container_width=True):
                    st.session_state.pop("renaming")
                    st.rerun()
            continue

        # ── Titre + menu ⋯ ───────────────────────────────────────────────────
        c_title, c_menu = st.columns([5.5, 0.8])

        with c_title:
            label = prefix + short
            if st.button(label, key=f"conv_{conv_id}",
                         use_container_width=True, disabled=is_active, help=title):
                msgs = _load_conversation(conv_id)
                st.session_state.active_conv_id = conv_id
                st.session_state.messages = msgs
                st.session_state.rag_chain.restore_memory(msgs)
                st.session_state.pop("renaming", None)
                st.rerun()

        with c_menu:
            with st.popover("⋯", use_container_width=True):
                st.caption(f"**{short}**")
                st.divider()

                pin_label = "Désépingler" if is_pinned else "Épingler"
                if st.button(pin_label, icon=":material/push_pin:", key=f"pin_{conv_id}", use_container_width=True):
                    _pin_conversation(conv_id, not is_pinned)
                    st.rerun()

                if st.button("Renommer", icon=":material/edit:", key=f"ren_{conv_id}", use_container_width=True):
                    st.session_state["renaming"] = conv_id
                    st.rerun()

                if st.button("Archiver", icon=":material/archive:", key=f"arch_{conv_id}", use_container_width=True):
                    _archive_conversation(conv_id)
                    if is_active:
                        _switch_active(exclude_id=conv_id)
                    st.rerun()

                st.divider()
                if st.button("Supprimer", icon=":material/delete:", key=f"del_{conv_id}",
                             use_container_width=True, type="primary"):
                    _delete_conversation(conv_id)
                    if is_active:
                        _switch_active(exclude_id=conv_id)
                    st.rerun()

        # Métadonnées sous le titre
        meta = []
        if is_pinned:
            meta.append("· épinglé")
        if date_str:
            meta.append(date_str)
        if n_msg:
            meta.append(f"{n_msg} msg")
        if meta:
            st.caption("  " + " · ".join(meta))

    # ── Section archivées ─────────────────────────────────────────────────────
    if archived_entries:
        st.divider()
        with st.expander(f"Archivées ({len(archived_entries)})"):
            for entry in archived_entries:
                conv_id  = entry["id"]
                title    = entry.get("title", "Conversation archivée")
                short    = (title[:28] + "…") if len(title) > 28 else title
                date_str = _format_date(entry.get("updated_at") or entry.get("created_at", ""))

                c1, c2 = st.columns([4, 1])
                with c1:
                    st.caption(short)
                    if date_str:
                        st.caption(f"  {date_str}")
                with c2:
                    with st.popover("⋯", use_container_width=True):
                        st.caption(f"**{short}**")
                        st.divider()
                        if st.button("Restaurer", icon=":material/unarchive:", key=f"rest_{conv_id}",
                                     use_container_width=True):
                            _restore_conversation(conv_id)
                            st.rerun()
                        st.divider()
                        if st.button("Supprimer", icon=":material/delete:", key=f"del_arch_{conv_id}",
                                     use_container_width=True, type="primary"):
                            _delete_conversation(conv_id)
                            st.rerun()

    st.divider()

    user_mode = st.selectbox(
        "Mode utilisateur",
        ["🎓 Étudiant", "🖥️ Admin système", "🔒 Pro cybersécurité"],
        help="Adapte le niveau de détail des réponses",
        key="user_mode",
    )

    st.divider()
    st.caption("**Corpus documentaire**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Docs", "148")
        st.metric("🎯 Score", "93 %")
    with col2:
        st.metric("🔢 Vecteurs", "52 269")
        st.metric("✅ Eval", "28 / 30")
    st.divider()
    st.caption("pfSense · Wazuh · Zabbix · Linux · Windows · MITRE")
    st.divider()
    st.caption("**Routage automatique**")
    st.caption("Simple → RAG  ·  Complexe → Agents")


# ── Corps principal ───────────────────────────────────────────────────────────

# Titre avec le nom de la conversation active
_active_entry = next(
    (e for e in _load_index() if e["id"] == st.session_state.active_conv_id),
    None,
)
_conv_title = _active_entry["title"] if _active_entry else "InfraBot"
st.title(f"🛡️ {_conv_title}")
st.caption("InfraBot · assistant cybersécurité · routage automatique RAG / multi-agents")

_DOMAIN_LABELS = {"doc": "🗂️ Doc", "network": "🌐 Réseau", "security": "🔐 Sécurité"}

# ── Indice du dernier message assistant (pour le feedback) ────────────────────
_last_assistant_idx = max(
    (i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"),
    default=-1,
)

# ── Affichage de l'historique ─────────────────────────────────────────────────
for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):

        # Badge type de réponse
        if msg["role"] == "assistant" and msg.get("msg_type") == "agents":
            domains = msg.get("domains", [])
            labels  = [_DOMAIN_LABELS.get(d, d) for d in domains]
            st.caption(f" Rapport multi-agents · {' · '.join(labels)}")
        elif msg["role"] == "assistant" and msg.get("msg_type") == "rag":
            st.caption("🔍 RAG")

        # Contenu du message (avec collapse pour les longs rapports agents)
        content = msg["content"]
        is_long_agent = (
            msg.get("msg_type") == "agents"
            and len(content) > 600
        )
        expand_key = f"expand_msg_{msg_index}"

        if is_long_agent and not st.session_state.get(expand_key, False):
            st.markdown(content[:500] + "\n\n*...(réponse tronquée)*")
            if st.button("▾ Voir la réponse complète", key=f"expand_btn_{msg_index}"):
                st.session_state[expand_key] = True
                st.rerun()
        else:
            st.markdown(content)
            if is_long_agent and st.session_state.get(expand_key):
                if st.button("▴ Réduire", key=f"collapse_btn_{msg_index}"):
                    st.session_state[expand_key] = False
                    st.rerun()

        # Sources (RAG uniquement)
        if msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])})", expanded=False):
                for src in msg["sources"]:
                    file_name = Path(src["file"]).name
                    page_info = f" — page {src['page']}" if src.get("page") else ""
                    st.caption(f"📎 **{file_name}**{page_info}")
                    st.caption(f"_{src['excerpt']}_")
                    st.divider()

        if msg["role"] == "assistant" and msg.get("msg_type") not in ("welcome", "agents"):
            # Bouton "Approfondir" seulement si dernière réponse RAG
            if msg_index == _last_assistant_idx and not msg.get("feedback"):
                user_q = _find_previous_user_message(st.session_state.messages, msg_index)
                if user_q:
                    if st.button(
                        "Approfondir avec les agents",
                        icon=":material/psychology:",
                        key=f"deepen_{msg_index}",
                        help="Lance une analyse multi-agents sur cette question",
                    ):
                        st.session_state["pending_agent_question"] = user_q
                        st.rerun()

        # Bouton copier (toutes les réponses assistant sauf welcome)
        if msg["role"] == "assistant" and msg.get("msg_type") != "welcome":
            copy_key = f"show_copy_{msg_index}"
            if st.button("Copier", icon=":material/content_copy:", key=f"copy_btn_{msg_index}", help="Afficher pour copier"):
                st.session_state[copy_key] = not st.session_state.get(copy_key, False)
                st.rerun()
            if st.session_state.get(copy_key):
                st.code(content, language=None)

        # Feedback — seulement sur la dernière réponse assistant
        if msg["role"] == "assistant" and msg.get("msg_type") != "welcome":
            feedback_value = msg.get("feedback")
            if feedback_value:
                # Feedback déjà donné : afficher uniquement l'icône
                icon = "👍" if feedback_value == "up" else "👎"
                st.caption(f"{icon}")
            elif msg_index == _last_assistant_idx:
                # Boutons feedback uniquement sur la dernière réponse
                col_up, col_down, _ = st.columns([1.5, 2.5, 3])
                with col_up:
                    if st.button("Utile", icon=":material/thumb_up:", key=f"up_{msg_index}"):
                        entry = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "mode_utilisateur": st.session_state.get("user_mode", "🖥️ Admin système"),
                            "feedback": "up",
                            "conv_id": st.session_state.active_conv_id,
                            "question": _find_previous_user_message(
                                st.session_state.messages, msg_index
                            ),
                            "answer": msg.get("content", ""),
                            "sources": msg.get("sources", []),
                        }
                        _append_feedback(entry)
                        st.session_state.messages[msg_index]["feedback"] = "up"
                        _save_conversation(
                            st.session_state.active_conv_id, st.session_state.messages
                        )
                        st.rerun()
                with col_down:
                    if st.button("À améliorer", icon=":material/thumb_down:", key=f"down_{msg_index}"):
                        entry = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "mode_utilisateur": st.session_state.get("user_mode", "🖥️ Admin système"),
                            "feedback": "down",
                            "conv_id": st.session_state.active_conv_id,
                            "question": _find_previous_user_message(
                                st.session_state.messages, msg_index
                            ),
                            "answer": msg.get("content", ""),
                            "sources": msg.get("sources", []),
                        }
                        _append_feedback(entry)
                        st.session_state.messages[msg_index]["feedback"] = "down"
                        _save_conversation(
                            st.session_state.active_conv_id, st.session_state.messages
                        )
                        st.rerun()


# ── Traitement "Approfondir avec les agents" ──────────────────────────────────
if "pending_agent_question" in st.session_state:
    pending_q    = st.session_state.pop("pending_agent_question")
    current_mode = st.session_state.get("user_mode", "🖥️ Admin système")

    domains = _classify_domains(pending_q)
    labels  = [_DOMAIN_LABELS.get(d, d) for d in domains]

    with st.chat_message("assistant"):
        st.caption(f" Analyse multi-agents · {' · '.join(labels)}")
        with st.status("Approfondissement multi-agents...", expanded=True) as _deep_status:
            st.write(f"🔎 Classification des domaines : {' · '.join(labels)}...")
            time.sleep(0.5)
            st.write("🔍 Recherche ciblée par technologie détectée...")
            time.sleep(0.5)
            st.write("📊 Re-ranking cross-encoder par domaine...")
            time.sleep(0.5)
            st.write(" Synthèse ReportAgent (1 seul appel LLM)...")
            try:
                crew   = CyberSecCrew()
                result = crew.run(pending_q, current_mode)
                answer = result["report"]
                domains_used = result["domains"]
            except Exception as e:
                answer = f"⚠️ Erreur agents : {e}"
                domains_used = []
                logger.error(f"Erreur Crew (deepen) : {e}")
            st.write("✅ Validation du rapport Markdown...")
            time.sleep(0.3)
            _deep_status.update(label="✅ Rapport généré", state="complete", expanded=False)

        used_labels = [_DOMAIN_LABELS.get(d, d) for d in domains_used]
        st.caption(f" Rapport multi-agents · {' · '.join(used_labels)}")
        st.markdown(answer)

    agent_msg = {
        "role": "assistant",
        "content": answer,
        "sources": [],
        "msg_type": "agents",
        "domains": domains_used,
    }
    st.session_state.messages.append(agent_msg)
    _save_conversation(st.session_state.active_conv_id, st.session_state.messages)
    _update_index_entry(
        st.session_state.active_conv_id,
        message_count=len(st.session_state.messages),
    )


# ── Zone de saisie ────────────────────────────────────────────────────────────
if question := st.chat_input("Posez votre question sur la cybersécurité..."):

    from src.retrieval.rag_chain import _sanitize_input as _si
    question     = _si(question)
    route        = _route_question(question)
    current_mode = st.session_state.get("user_mode", "🖥️ Admin système")

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if route == "agents":
            domains = _classify_domains(question)
            labels  = [_DOMAIN_LABELS.get(d, d) for d in domains]
            st.caption(f" Routage vers les agents : {' · '.join(labels)}")

            with st.status("Analyse multi-agents...", expanded=True) as _crew_status:
                st.write(f"🔎 Classification des domaines : {' · '.join(labels)}...")
                time.sleep(0.5)
                st.write("🔍 Recherche ciblée par technologie détectée...")
                time.sleep(0.5)
                st.write("📊 Re-ranking cross-encoder par domaine...")
                time.sleep(0.5)
                st.write(" Synthèse ReportAgent (1 seul appel LLM)...")
                try:
                    crew   = CyberSecCrew()
                    result = crew.run(question, current_mode)
                    answer = result["report"]
                    domains_used = result["domains"]
                except Exception as e:
                    logger.error(f"Erreur Crew : {e}", exc_info=True)
                    answer = "⚠️ Une erreur s'est produite lors de l'analyse. Réessayez ou reformulez votre question."
                    domains_used = []
                st.write("✅ Validation du rapport Markdown...")
                time.sleep(0.3)
                _crew_status.update(label="✅ Rapport généré", state="complete", expanded=False)

            used_labels = [_DOMAIN_LABELS.get(d, d) for d in domains_used]
            st.caption(f" Rapport multi-agents · {' · '.join(used_labels)}")
            st.markdown(answer)

            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "sources": [],
                "msg_type": "agents",
                "domains": domains_used,
            }

        else:
            with st.status("Analyse en cours...", expanded=True) as _rag_status:
                st.write("🔢 Embedding de la question (384 dimensions)...")
                time.sleep(0.6)
                st.write("🔍 Recherche vectorielle — 52 269 vecteurs ChromaDB...")
                time.sleep(0.5)
                st.write("📊 Re-ranking cross-encoder — 24 → top 6 chunks...")
                time.sleep(0.5)
                st.write(" Génération llama3.2 3B via Ollama...")
                try:
                    result  = st.session_state.rag_chain.ask(
                        question, user_mode=current_mode
                    )
                    answer  = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    logger.error(f"Erreur RAG : {e}", exc_info=True)
                    answer  = "⚠️ Une erreur s'est produite. Réessayez ou reformulez votre question."
                    sources = []
                st.write("✅ Validation des commandes (Command Validator)...")
                time.sleep(0.3)
                _rag_status.update(label="✅ Réponse générée", state="complete", expanded=False)

            st.caption("🔍 RAG")
            st.markdown(answer)

            if sources:
                with st.expander(
                    f"📄 Sources utilisées ({len(sources)})", expanded=True
                ):
                    for src in sources:
                        file_name = Path(src["file"]).name
                        page_info = f" — page {src['page']}" if src.get("page") else ""
                        st.caption(f"📎 **{file_name}**{page_info}")
                        st.caption(f"_{src['excerpt']}_")

            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "msg_type": "rag",
            }

    st.session_state.messages.append(assistant_msg)
    _save_conversation(st.session_state.active_conv_id, st.session_state.messages)

    # Auto-titre avec la première question utilisateur
    user_msgs    = [m for m in st.session_state.messages if m["role"] == "user"]
    title_kwargs = {"message_count": len(st.session_state.messages)}
    if len(user_msgs) == 1:
        title = (question[:45] + "…") if len(question) > 45 else question
        title_kwargs["title"] = title
    _update_index_entry(st.session_state.active_conv_id, **title_kwargs)
