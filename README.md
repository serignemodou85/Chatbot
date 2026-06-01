# InfraBot — Assistant IA Cybersécurité & Infrastructure

Chatbot IA spécialisé en cybersécurité, infrastructure réseau et administration système.

- **Phase 1** — RAG conversationnel : LangChain + ChromaDB + llama3.2 + Streamlit
- **Phase 2** — Multi-agents : CrewAI + retrieval Python ciblé par outil + agent de synthèse

---

## Table des matières

1. [Prérequis](#1-prérequis)
2. [Installation — première fois](#2-installation--première-fois)
3. [Configuration (.env)](#3-configuration-env)
4. [Indexer les documents](#4-indexer-les-documents)
5. [Lancer le chatbot](#5-lancer-le-chatbot)
6. [Phase 2 — Agents CrewAI](#6-phase-2--agents-crewai)
7. [Lancer avec Docker](#7-lancer-avec-docker)
8. [Tests et évaluation](#8-tests-et-évaluation)
9. [Maintenance](#9-maintenance)
10. [Structure du projet](#10-structure-du-projet)
11. [Résolution de problèmes](#11-résolution-de-problèmes)

---

## 1. Prérequis

| Outil | Version | Pourquoi | Vérification |
|-------|---------|----------|--------------|
| Python | **3.12.x** | Runtime du projet | `python --version` |
| Git | ≥ 2.x | Versionnement | `git --version` |
| Ollama | ≥ 0.23 | LLM local llama3.2 (gratuit) | `ollama --version` |
| Docker Desktop | ≥ 29 | Déploiement conteneurisé | `docker --version` |

> **Note Windows** : Git for Windows s'installe dans `C:\Program Files\Git\`.
> Si `git` n'est pas reconnu dans PowerShell, ajouter `C:\Program Files\Git\cmd` au PATH.

---

## 2. Installation — première fois

### Étape 1 — Cloner le projet

```powershell
git clone https://github.com/tello/rag-chatbot.git
cd rag_chatbot
```

### Étape 2 — Créer l'environnement virtuel Python

```powershell
python -m venv venv
venv\Scripts\activate
python --version   # doit afficher Python 3.12.x
```

### Étape 3 — Installer les dépendances Python

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> La première installation prend ~5 minutes (PyTorch, sentence-transformers, chromadb...).

### Étape 4 — Installer Ollama + télécharger llama3.2

```powershell
# 1. Télécharger depuis https://ollama.com → Windows (OllamaSetup.exe)

# 2. Vérifier
ollama --version

# 3. Télécharger llama3.2 (3B params, ~2 GB — rapide sur CPU)
ollama pull llama3.2

# 4. Tester
ollama run llama3.2 "Réponds juste 'ok' pour tester"
```

> **Si ollama n'est pas reconnu dans PowerShell** :
> Ajouter `C:\Users\<ton_user>\AppData\Local\Programs\Ollama` au PATH système.

---

## 3. Configuration (.env)

```powershell
copy .env.example .env
```

Ouvrir `.env` et vérifier :

```ini
# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2          # 3B params, ~2 Go RAM, ~3-5 min/réponse sur CPU
LLM_TEMPERATURE=0.1
OPENAI_API_KEY=              # Laisser vide si Ollama
OLLAMA_BASE_URL=http://localhost:11434

# ── Phase 2 — Agents CrewAI ──────────────────────────────────────────────────
CREW_LLM_MODEL=llama3.2     # même modèle : cohérence + mémoire RAM partagée

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION=cybersec_docs

# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RETRIEVER_K=6
```

> `.env` est dans `.gitignore` — il ne sera jamais commité sur GitHub.

---

## 4. Indexer les documents

L'indexation convertit tes PDFs/docs en vecteurs stockés dans ChromaDB.
**À faire une seule fois** (ou après ajout de nouveaux documents).

### Placer les documents

```
data/docs/
├── wazuh/          ← Guide installation, agent, analyse, POC Lab, Ruleset
├── zabbix/         ← Documentation 7.4, hôtes, installation
├── dolibarr/       ← Installation, sécurité, organisation
├── Ubuntu Server/  ← Guide Ubuntu Server
├── linux/          ← Admin Linux, System Administration
├── pfsense/        ← Documentation Netgate, OpenVPN
└── Windows Server/ ← Installation/Configuration, Admin Guide
```

Formats supportés : `.pdf`, `.docx`, `.txt`, `.md`, `.html`

### Lancer l'indexation

```powershell
venv\Scripts\activate

# Indexation complète (première fois ou après ajout de docs)
python scripts/build_index.py --reset

# Ajouter des docs sans écraser l'existant
python scripts/build_index.py

# Options
python scripts/build_index.py --docs-dir C:\mes\docs
python scripts/build_index.py --dedupe-threshold 0.95
```

**Sortie attendue :**
```
INFO  Fichiers trouvés : 21
INFO  Chargé : Wazuh - Deployment and Administration Guide.pdf (55 sections)
INFO  Total chunks produits : 29 897
INFO  Indexation de 29 897 chunks en 6 batch(es) dans ChromaDB...
INFO  Base vectorielle créée : 29 897 vecteurs stockés dans ./data/chroma_db
INFO  Indexation terminée en ~796s
```

> ⚠️ Ne pas lancer `build_index.py` sans `--reset` si la base existe déjà,
> sous peine de doubler les vecteurs (doublons = qualité de recherche réduite).

---

## 5. Lancer le chatbot

```powershell
venv\Scripts\activate

# Sur Windows (contourne Device Guard qui bloque les .exe en répertoire utilisateur)
run.bat

# Ou directement
venv\Scripts\python -m streamlit run src/interface/app.py
```

Ouvrir **http://localhost:8501** dans le navigateur.

### Interface

L'interface est un **chat unifié avec routage automatique** :

- **Questions simples** (définition, procédure single-outil) → RAG direct (~30 s - 2 min)
- **Questions complexes** (comparaisons, multi-outils, déploiement) → Agents CrewAI (~2-5 min)

```
┌──────────────────────────────────────────────────────────────┐
│ SIDEBAR                │ 🛡️ InfraBot                         │
│                        │                                      │
│ Mode utilisateur :     │ 💬 Chat unifié                       │
│  🎓 Étudiant           │                                      │
│  🖥️ Admin système      │ [Historique messages]                │
│  🔒 Pro cybersécurité  │ [Sources cliquables]                 │
│                        │ [👍 / 👎 feedback]                   │
│ [Nouvelle conv.]       │                                      │
│ [Conversations]        │ [Saisie question...]                 │
│                        │                                      │
└──────────────────────────────────────────────────────────────┘
```

### Icônes Material et améliorations UX

L'interface utilise les **icônes Material Design** de Streamlit 1.40+ :

| Élément | Icône Material |
|---------|---------------|
| Nouvelle conversation | `:material/add:` |
| Épingler / Désépingler | `:material/push_pin:` |
| Renommer | `:material/edit:` |
| Archiver / Restaurer | `:material/archive:` / `:material/unarchive:` |
| Supprimer | `:material/delete:` |
| Sauver / Annuler le renommage | `:material/check:` / `:material/close:` |
| Approfondir avec les agents | `:material/psychology:` |
| Copier la réponse | `:material/content_copy:` |
| Feedback utile / à améliorer | `:material/thumb_up:` / `:material/thumb_down:` |

CSS injecté au chargement :
- Conversation active = fond violet + bordure gauche (`:material/push_pin:` CSS)
- Hover discret sur les autres conversations
- Boutons feedback compacts en forme de pilule

### Exemples de questions

```
🎓 Mode Étudiant :
  "C'est quoi Wazuh ?"
  "Comment fonctionne un SIEM ?"
  "Explique-moi la supervision réseau"

🖥️ Mode Admin système :
  "Comment installer l'agent Wazuh sur Ubuntu ?"
  "Comment ajouter un hôte dans Zabbix ?"
  "Quelle commande pour vérifier les logs Wazuh ?"

🔒 Mode Pro cybersécurité :
  "Quelles sont les limites de Wazuh pour la détection d'intrusions ?"
  "Comment configurer les règles de corrélation dans Wazuh ?"
  "Analyse des risques liés à Dolibarr exposé sur Internet"
```

---

## 6. Phase 2 — Agents CrewAI

La Phase 2 mobilise des agents spécialisés pour les questions complexes.
Le routage est **automatique** — pas besoin de choisir manuellement.

### Architecture

```
Question complexe
       │
       ▼
_classify(question)          ← Python, ~0 ms, pas de LLM
       │
  ┌────┴────┐
  ▼         ▼
network?  security?  doc?    ← mots-clés (_NETWORK_KW, _SECURITY_KW)
  │         │         │
  ▼         ▼         ▼
Recherche ciblée par outil dans ChromaDB
  pfsense   wazuh    tous
  openvpn   zabbix
  │         │         │
  └────┬────┘─────────┘
       ▼
Re-ranking cross-encoder     ← réordonne par vraie pertinence
       │
       ▼
  Agent Rapport               ← un seul appel LLM pour la synthèse
       │
       ▼
  Rapport final structuré Markdown
```

### Le routage automatique

| Signal | Exemple | Route |
|--------|---------|-------|
| Question courte, 1 outil | "C'est quoi Wazuh ?" | RAG direct |
| Comparaison multi-outils | "Différence Wazuh vs Zabbix ?" | Agents |
| Déploiement d'infrastructure | "Déployer pfSense + Wazuh pour 50 postes" | Agents |
| Mot-clé agent explicite | "compare", "versus", "infrastructure" | Agents |

### Tester les agents en CLI

```powershell
venv\Scripts\activate
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

# Tester le routage uniquement (rapide)
python scripts/run_crew.py --classify-only --question "Compare Wazuh et Zabbix"
# Sortie : Routage : ['security']

# Lancer un crew complet (~2-5 min avec llama3.2)
python scripts/run_crew.py --question "Compare Wazuh et Zabbix" --mode admin

# Options
python scripts/run_crew.py --mode etudiant
python scripts/run_crew.py --mode pro --question "Analyse CVE Zabbix 2024"
```

### Command Validator — filtrage automatique des commandes inventées

Les LLMs locaux (llama3.2 3B) peuvent générer des commandes inexistantes même avec un prompt strict.
Exemple observé : question SSH → `swanctl` (IPsec, hors sujet), `enablesshd` (commande inventée).

Chaque réponse passe par `src/retrieval/command_validator.py` **après** génération :

```
LLM génère réponse
        ↓
Niveau 1 — Mélange de technologies
  swanctl (IPsec) + sshd (SSH serveur) dans la même réponse → bloqué
  pfctl (BSD/pfSense) + iptables (Linux) → OS incompatibles → bloqué
        ↓
Niveau 2 — Whitelist par technologie détectée
  pfSense → GUI uniquement, 0 commande CLI autorisée
  strongSwan → uniquement "swanctl"
  FreeBSD pf → uniquement "pfctl"
  Linux → liste étendue (ip, iptables, nft, systemctl, apt…)
        ↓
Niveau 3 — Présence dans le contexte RAG (fallback)
  Toute commande doit apparaître textuellement dans les chunks récupérés
  Commande absente → réponse entière remplacée par un message d'absence
```

**Patterns de détection des commandes dans la réponse :**
- Blocs de code ` ``` ``` ` et inline `` `commande` ``
- Lignes shell `$`, `#`, `>`
- Tokens `*ctl` (swanctl, systemctl, pfctl…)
- Chemins Unix absolus `/bin/`, `/etc/`…
- `sudo <commande>`
- CamelCase technique (`SSHdKeyOnly`, `PublicKeyOnly`)
- `enable*` composés (`enablesshd`, `enablefirewall`)

**Logs produits :**
```
WARNING [CommandValidator] Commandes absentes du contexte RAG : ['enablesshd', 'sshdkeyonly'] — réponse bloquée.
WARNING [CommandValidator] pfSense (GUI only) — 2 commande(s) CLI bloquée(s) : ['pfctl', 'iptables']
WARNING [CommandValidator] Mélange technologique — familles : ['ipsec', 'ssh_srv'] — réponse bloquée.
```

### Exemples de questions pour les agents

```
🌐 Réseau :
  "Comment configurer un VPN site-à-site avec pfSense ?"
  "Explique les VLANs et leur utilité dans un réseau d'entreprise"

🔐 Sécurité :
  "Comment configurer la détection SSH brute-force dans Wazuh ?"
  "Configure les alertes Zabbix pour détecter une charge CPU anormale"

🌐🔐 Multi-outils :
  "Je veux déployer pfSense + Wazuh + Zabbix pour 50 postes. Par où commencer ?"
  "Compare Wazuh et Zabbix pour superviser 50 machines"
```

---

## 7. Lancer avec Docker

```powershell
# 1. Construire l'image (~5-10 min première fois)
docker build -t infrabot .

# 2. Lancer tous les services
docker compose up -d

# 3. Vérifier
docker compose ps

# 4. Indexer les documents (première fois)
docker compose run --rm indexer

# 5. Accéder au chatbot
# http://localhost:8501
```

### Commandes Docker du quotidien

```powershell
docker compose down                   # Arrêter
docker compose restart app            # Redémarrer l'app
docker compose build app && docker compose up -d app   # Après modif du code
docker compose exec app bash          # Shell dans le container
docker compose logs -f app            # Logs en temps réel
docker stats                          # Consommation ressources
```

### Premier lancement Ollama dans Docker

```powershell
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2
docker compose up -d app
```

---

## 8. Tests et évaluation

### Tests automatisés

```powershell
venv\Scripts\activate
pytest tests/ -v                      # Tous les tests
pytest tests/test_pipeline.py -v     # Pipeline RAG uniquement
pytest tests/test_phase2.py -v       # Phase 2 agents
```

**Tests couverts :**
- Chunking et métadonnées
- Build/load ChromaDB
- Recherche par similarité + re-ranking
- Validation du prompt RAG (anti-hallucination)
- Déduplication
- Routage des agents

### Évaluation de la qualité RAG

```powershell
python scripts/evaluate_rag.py
python scripts/evaluate_rag.py --verbose
python scripts/evaluate_rag.py --target 0.75
python scripts/evaluate_rag.py --mode "🔒 Pro cybersécurité" --verbose
```

**Métriques :**

| Métrique | Description |
|---------|-------------|
| `keyword_ratio` | % des mots-clés attendus trouvés dans la réponse |
| `source_relevance` | Les sources citées sont-elles pertinentes ? |
| `has_substantive_answer` | Réponse réelle (pas un refus) |
| `latency_s` | Temps de réponse en secondes |
| `sources_count` | Nombre de sources distinctes citées |

---

## 9. Maintenance

### Ajouter de nouveaux documents

```powershell
# 1. Copier les PDFs dans data/docs/
copy "C:\mes\docs\*.pdf" data\docs\wazuh\

# 2. Réindexer proprement depuis zéro
python scripts/build_index.py --reset
```

> Toujours utiliser `--reset` quand on réindexe après ajout de docs.
> Sans `--reset`, les anciens vecteurs restent et se cumulent avec les nouveaux (doublons).

### Supprimer les doublons dans ChromaDB

```powershell
python scripts/deduplicate_chroma.py --dry-run    # Aperçu
python scripts/deduplicate_chroma.py              # Supprimer
python scripts/deduplicate_chroma.py --threshold 0.95
```

### Réinitialiser complètement

```powershell
python scripts/build_index.py --reset
```

---

## 10. Structure du projet

```
rag_chatbot/
├── .env                    ← Secrets (jamais dans git)
├── .env.example            ← Template de configuration
├── .gitignore              ← Exclut data/, venv/, .env
├── .gitattributes          ← Normalise LF/CRLF
├── Dockerfile              ← Image Docker Python 3.12
├── docker-compose.yml      ← Services : app + ollama + indexer
├── requirements.txt        ← Dépendances Python (Phase 1 + 2)
├── run.bat                 ← Lancement Windows (contourne Device Guard)
├── cours.md                ← Guide d'apprentissage de la stack
│
├── config/
│   └── settings.py         ← Chargement centralisé du .env
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py  ← Chunking hiérarchique + récursif
│   │   ├── vectorstore.py      ← ChromaDB (batch 5000, MMR + RerankedRetriever)
│   │   └── deduplication.py    ← Anti-doublons (seuil 0.92)
│   ├── llm/
│   │   └── llm_factory.py      ← Factory OpenAI / Ollama
│   ├── retrieval/
│   │   ├── rag_chain.py        ← Pipeline RAG conversationnel (llama3.2)
│   │   ├── reranker.py         ← Cross-encoder re-ranking (ms-marco-MiniLM-L-12-v2)
│   │   ├── cache.py            ← Cache sémantique LRU (cosine ≥ 0.95)
│   │   └── command_validator.py← Post-processing anti-hallucination (3 niveaux)
│   ├── agents/                 ← Phase 2 — Multi-agents CrewAI
│   │   ├── tools/
│   │   │   └── rag_tools.py    ← 3 outils ChromaDB (doc / réseau / sécurité)
│   │   ├── agents.py           ← Factory : doc / network / security / report
│   │   ├── tasks.py            ← make_research_task + make_report_task
│   │   └── crew.py             ← _classify() + retrieval Python + CyberSecCrew.run()
│   └── interface/
│       └── app.py              ← Streamlit : chat unifié + routage automatique
│
├── scripts/
│   ├── build_index.py          ← Indexation (--reset recommandé)
│   ├── deduplicate_chroma.py   ← Nettoyage doublons
│   ├── evaluate_rag.py         ← Évaluation qualité Phase 1
│   └── run_crew.py             ← Test CLI du crew Phase 2
│
├── tests/
│   ├── test_pipeline.py        ← Tests intégration pipeline RAG
│   ├── test_quality_tools.py   ← Tests déduplication et modes
│   └── test_phase2.py          ← Tests agents Phase 2
│
└── data/                       ← Ignoré par git
    ├── docs/                   ← 21 documents source (PDFs)
    ├── chroma_db/              ← Base vectorielle (~29 897 vecteurs)
    ├── embedding_cache/        ← Modèle HuggingFace all-MiniLM-L6-v2 (cache)
    ├── feedback/               ← Feedbacks utilisateurs (JSON)
    ├── session/                ← Historique conversations (JSON)
    └── eval/                   ← Rapports d'évaluation
```

---

## 11. Résolution de problèmes

### `ollama` non reconnu dans PowerShell

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\Ollama;" + $env:PATH
ollama --version
```

### Erreur `Base ChromaDB introuvable`

```powershell
python scripts/build_index.py --reset
```

### Ollama ne répond pas

```powershell
curl http://localhost:11434/api/tags
ollama serve        # si pas de réponse
Get-Process ollama  # vérifier le processus
```

### Modèle llama3.2 non trouvé

```powershell
ollama list
ollama pull llama3.2
ollama run llama3.2 "test"
```

### Réponses très lentes (> 10 min)

La cause la plus fréquente : mauvais modèle dans `.env`.

```ini
# .env — vérifier que c'est bien llama3.2 (3B) et non llama3 (8B)
LLM_MODEL=llama3.2
CREW_LLM_MODEL=llama3.2
```

Temps attendus sur CPU (AMD Ryzen, pas de GPU) :
- RAG simple : ~2-3 min
- Agents multi-outils : ~3-5 min

### Device Guard bloque `streamlit.exe` (Windows)

```powershell
# Utiliser run.bat à la place de streamlit directement
run.bat
# ou
venv\Scripts\python -m streamlit run src/interface/app.py
```

### `ModuleNotFoundError` au lancement

```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

### ChromaDB — doublons (vecteurs > 2× le nombre de chunks)

```powershell
# Réindexer proprement
python scripts/build_index.py --reset
```

### ChromaDB — `sqlite3.OperationalError: no such column: collections.topic`

```powershell
$env:PYTHONPATH = "."
venv\Scripts\python scripts/_fix_chroma_schema.py
# ou relancer build_index.py — le fix est appliqué automatiquement
```

### Agents CrewAI — `UnicodeEncodeError` dans le terminal

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/run_crew.py --question "..."
```

### Qualité des réponses insuffisante

```powershell
# Vérifier le nombre de chunks indexés
python -c "
import sys; sys.path.insert(0,'.')
from src.ingestion.vectorstore import VectorStoreManager
vs = VectorStoreManager(); vs.load()
print('Chunks:', vs._vectorstore._collection.count())
"
# Doit afficher ~29 897. Si > 50 000 : doublons → relancer build_index.py --reset
```

---

## Versions vérifiées

| Composant | Version |
|-----------|---------|
| Python | 3.12.9 |
| langchain | 0.1.20 |
| langchain-core | 0.1.53 |
| chromadb | 0.4.24 |
| crewai | 0.28.8 |
| sentence-transformers | 3.0.1 |
| streamlit | 1.58.0 |
| Ollama | 0.23.3 |
| Docker | 29.3.0 |

> **Contrainte de compatibilité** : crewai 0.28.8 est la dernière version compatible avec langchain 0.1.x.
> crewai ≥ 0.51 requiert langchain ≥ 0.3 (upgrade majeur, non prévu en Phase 2).

---

## Choix d'architecture — Phase 2

### Pourquoi Python RAG + 1 agent au lieu de 4 agents LLM actifs ?

La vision originale du multi-agents est la suivante : chaque question est traitée par l'agent qui la concerne (Agent Réseau pour pfSense/VPN, Agent Sécurité pour Wazuh/Zabbix, etc.), chacun raisonnant sur sa recherche avant de passer le résultat à un Agent Rapport pour la synthèse finale. C'est architecturalement correct.

**La contrainte réelle : LLM local sur CPU.**

| Architecture | Appels LLM | Temps sur CPU (llama3.2) |
|-------------|-----------|--------------------------|
| 4 agents LLM actifs | 4 à 8 appels | 20 à 40 minutes |
| Python RAG + 1 agent | 1 appel | 3 à 5 minutes |

Avec 4 agents actifs, une question complexe prendrait 20 à 40 minutes — inutilisable en pratique. La spécialisation est donc assurée par du code Python déterministe (filtrage par mots-clés, recherche ciblée par outil dans ChromaDB, re-ranking), et le LLM n'intervient qu'une seule fois pour la synthèse finale.

**La qualité du contexte donné au LLM est identique** : qu'un agent LLM ou du code Python sélectionne les chunks Wazuh, le LLM de synthèse reçoit les mêmes extraits documentaires.

**Ce choix est réversible** : l'architecture est conçue pour que le passage à 4 agents LLM actifs se fasse en remplaçant le retrieval Python par des `CrewAI Agent` avec outils — sans toucher à l'interface ni à la base vectorielle. Il suffit de disposer d'un GPU ou d'une API LLM (GPT-4, Claude) pour que ce soit viable.

---

## Roadmap

- ~~**Phase 1**~~ ✅ RAG Chat (LangChain + ChromaDB + llama3.2 + Streamlit)
- ~~**Phase 2**~~ ✅ Agents spécialisés CrewAI (routage auto + retrieval ciblé + re-ranking)
- ~~**Phase 2.5**~~ ✅ Qualité & UX — Command Validator (anti-hallucination 3 niveaux) + UI Material icons
- **Phase 3** — Audit automatisé (Nmap, OWASP ZAP, MobSF) + rapports automatiques

---

> **Nom du projet : InfraBot**
