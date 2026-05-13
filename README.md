# CyberSec RAG Chatbot — Phase 1

Chatbot IA spécialisé en cybersécurité, infrastructure réseau et administration système.
Architecture RAG (Retrieval-Augmented Generation) : LangChain + ChromaDB + Llama 3 (ou GPT-4o) + Streamlit.

---

## Table des matières

1. [Prérequis](#1-prérequis)
2. [Installation — première fois](#2-installation--première-fois)
3. [Configuration (.env)](#3-configuration-env)
4. [Indexer les documents](#4-indexer-les-documents)
5. [Lancer le chatbot](#5-lancer-le-chatbot)
6. [Lancer avec Docker](#6-lancer-avec-docker)
7. [Tests et évaluation](#7-tests-et-évaluation)
8. [Maintenance](#8-maintenance)
9. [Structure du projet](#9-structure-du-projet)
10. [Résolution de problèmes](#10-résolution-de-problèmes)

---

## 1. Prérequis

| Outil | Version | Pourquoi | Vérification |
|-------|---------|----------|--------------|
| Python | **3.12.x** | Runtime du projet | `python --version` |
| Git | ≥ 2.x | Versionnement | `git --version` |
| Ollama | ≥ 0.23 | LLM local Llama 3 (gratuit) | `ollama --version` |
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
# Créer le venv (une seule fois)
python -m venv venv

# Activer (Windows PowerShell)
venv\Scripts\activate

# Vérifier que le bon Python est utilisé
python --version   # doit afficher Python 3.12.x
```

### Étape 3 — Installer les dépendances Python

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> La première installation prend ~5 minutes (PyTorch, sentence-transformers, chromadb...).

### Étape 4 — Installer Ollama + télécharger Llama 3

```powershell
# 1. Télécharger l'installeur depuis https://ollama.com → Windows
#    Lancer OllamaSetup.exe et suivre l'assistant

# 2. Vérifier l'installation
ollama --version   # ollama version is 0.x.x

# 3. Démarrer le serveur Ollama (se lance automatiquement au démarrage Windows)
ollama serve       # dans un terminal séparé, ou laisser tourner en tâche de fond

# 4. Télécharger le modèle Llama 3 (~4.7 GB, une seule fois)
ollama pull llama3

# 5. Tester que le modèle répond
ollama run llama3 "Réponds juste 'ok' pour tester"
```

> **Si ollama n'est pas reconnu dans PowerShell** :
> Ajouter `C:\Users\<ton_user>\AppData\Local\Programs\Ollama` au PATH système.

---

## 3. Configuration (.env)

```powershell
# Copier le template
copy .env.example .env
```

Ouvrir `.env` et vérifier/ajuster :

```ini
# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER=ollama          # "ollama" (local gratuit) ou "openai" (payant)
LLM_MODEL=llama3             # "llama3" si Ollama, "gpt-4o" si OpenAI
LLM_TEMPERATURE=0.1          # 0=déterministe, 1=créatif

OPENAI_API_KEY=              # Laisser vide si Ollama
OLLAMA_BASE_URL=http://localhost:11434

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=huggingface              # Local, gratuit, aucune clé requise
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION=cybersec_docs

# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_SIZE=512               # Tokens par chunk
CHUNK_OVERLAP=64             # Chevauchement entre chunks consécutifs
RETRIEVER_K=4                # Nombre de chunks retournés par requête
```

> `.env` est dans `.gitignore` — il ne sera jamais commité sur GitHub.

---

## 4. Indexer les documents

L'indexation convertit tes PDFs/docs en vecteurs stockés dans ChromaDB.
**À faire une seule fois** (ou après ajout de nouveaux documents).

### Placer les documents

```
data/docs/
├── wazuh/          ← PDFs Wazuh (agent, installation, analyse...)
├── zabbix/         ← PDFs Zabbix (hôtes, installation, doc 7.4...)
├── dolibarr/       ← PDFs Dolibarr (install, orga, sécurité...)
├── Ubuntu Server/  ← Guide Ubuntu Server
├── linux/          ← Tes docs Linux (à ajouter)
├── pfsense/        ← Docs pfSense/OpenVPN (à ajouter)
└── Windows Server/ ← Docs Windows Server (à ajouter)
```

Formats supportés : `.pdf`, `.docx`, `.txt`, `.md`, `.html`

### Lancer l'indexation

```powershell
# Venv activé obligatoire
venv\Scripts\activate

# Indexation standard
python scripts/build_index.py

# Options avancées
python scripts/build_index.py --reset                    # Réindexer depuis zéro
python scripts/build_index.py --docs-dir C:\mes\docs     # Dossier personnalisé
python scripts/build_index.py --dedupe-threshold 0.95    # Seuil anti-doublons
```

**Sortie attendue :**
```
INFO  Fichiers trouvés : 12
INFO  Chargé : Zabbix_Documentation_7.4.en.pdf (2239 sections)
INFO  Total chunks produits : 11490
INFO  Indexation de 11490 chunks dans ChromaDB...
INFO  Base vectorielle créée : 11490 vecteurs stockés dans ./data/chroma_db
```

> La première fois : ~10-15 min (téléchargement modèle HuggingFace ~90 MB + vectorisation).
> Les fois suivantes : ~2-3 min (modèle déjà en cache dans `data/embedding_cache/`).

---

## 5. Lancer le chatbot

```powershell
# Prérequis : venv activé + index ChromaDB construit + Ollama serve lancé
venv\Scripts\activate

# Lancer l'interface
streamlit run src/interface/app.py
```

Ouvrir **http://localhost:8501** dans le navigateur.

### Interface

```
┌──────────────────────────────────────────────────┐
│ SIDEBAR                │ CHAT                    │
│                        │                         │
│ Mode utilisateur :     │ 🛡️ CyberSec RAG Chatbot │
│  🎓 Étudiant           │                         │
│  🖥️ Admin système      │ [Historique messages]   │
│  🔒 Pro cybersécurité  │                         │
│                        │ [Sources utilisées]     │
│ [Nouvelle conv.]       │                         │
│                        │ [👍 Utile / 👎 À amélio.]│
│ Base documentaire :    │                         │
│  Wazuh · Zabbix        │ [ Posez votre question ]│
│  Dolibarr · Ubuntu     │                         │
└──────────────────────────────────────────────────┘
```

### Exemples de questions à poser

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

## 6. Lancer avec Docker

Docker empaquette tout (Python, dépendances, code) dans une image portable.

### Prérequis

- Docker Desktop installé et démarré
- `.env` configuré (même fichier qu'en local)

### Démarrage complet

```powershell
# 1. Construire l'image (première fois : ~5-10 min)
docker build -t rag-chatbot .

# 2. Lancer tous les services (app + Ollama)
docker compose up -d

# 3. Vérifier que tout tourne
docker compose ps

# 4. Indexer les documents dans Docker (première fois seulement)
docker compose run --rm indexer

# 5. Voir les logs en temps réel
docker compose logs -f app

# 6. Accéder au chatbot
# Ouvrir http://localhost:8501
```

### Commandes Docker du quotidien

```powershell
# Arrêter tout
docker compose down

# Redémarrer uniquement l'app
docker compose restart app

# Mettre à jour après modification du code
docker compose build app
docker compose up -d app

# Ouvrir un shell dans le container (debug)
docker compose exec app bash

# Voir la consommation de ressources
docker stats
```

### Premier lancement avec Ollama dans Docker

```powershell
# 1. Démarrer Ollama seulement
docker compose up -d ollama

# 2. Télécharger le modèle Llama 3 dans le container
docker compose exec ollama ollama pull llama3

# 3. Démarrer l'application
docker compose up -d app
```

---

## 7. Tests et évaluation

### Tests automatisés (unitaires + intégration)

```powershell
venv\Scripts\activate
pytest tests/ -v                          # Tous les tests
pytest tests/test_pipeline.py -v          # Pipeline uniquement
pytest tests/ -v -k "test_chunking"       # Filtre par nom
```

**Tests couverts :**
- Chunking et métadonnées des documents
- Build/load ChromaDB
- Recherche par similarité
- Validation du prompt RAG
- Algorithme de déduplication

### Évaluation de la qualité RAG

```powershell
# Évaluation standard sur le jeu de test (data/eval/test_questions.json)
python scripts/evaluate_rag.py

# Avec affichage des réponses
python scripts/evaluate_rag.py --verbose

# Avec objectif personnalisé
python scripts/evaluate_rag.py --target 0.75

# En mode Pro cybersécurité
python scripts/evaluate_rag.py --mode "🔒 Pro cybersécurité" --verbose
```

**Métriques mesurées :**

| Métrique | Description |
|---------|-------------|
| `keyword_ratio` | % des mots-clés attendus trouvés dans la réponse |
| `source_relevance` | Les sources citées sont-elles pertinentes au sujet ? |
| `has_substantive_answer` | La réponse contient du contenu réel (pas un refus) |
| `latency_s` | Temps de réponse en secondes |
| `sources_count` | Nombre de sources distinctes citées |

**Sortie attendue :**
```
────────────────────────────────────────────────────────────
  RAPPORT D'ÉVALUATION RAG
────────────────────────────────────────────────────────────
  Questions testées : 4
  Réponses pertinentes : 3/4
  Taux de pertinence : 75.0%
  Latence moyenne : 8.32s
  Objectif (80%) atteint : ✗ NON
────────────────────────────────────────────────────────────
  [✓] Q1: A quoi sert Wazuh dans une architecture securite ?
       Mots-clés  : 80% (4/5 — trouvés: ['wazuh', 'siem', 'detection', 'incidents'])
       Sources    : 3 citées, pertinence: 100%
       Latence    : 7.45s
```

---

## 8. Maintenance

### Ajouter de nouveaux documents

```powershell
# 1. Copier les nouveaux PDFs dans data/docs/
copy "C:\mes\nouveaux\docs\*.pdf" data\docs\linux\

# 2. Réindexer (ajoute sans écraser l'existant)
python scripts/build_index.py

# Ou réindexer depuis zéro si besoin
python scripts/build_index.py --reset
```

### Supprimer les doublons dans ChromaDB

```powershell
# Aperçu (sans modifier)
python scripts/deduplicate_chroma.py --dry-run

# Supprimer les doublons (seuil 0.92 par défaut)
python scripts/deduplicate_chroma.py

# Avec seuil personnalisé
python scripts/deduplicate_chroma.py --threshold 0.95
```

### Réinitialiser complètement la base

```powershell
# Supprimer ChromaDB et réindexer
python scripts/build_index.py --reset
```

---

## 9. Structure du projet

```
rag_chatbot/
├── .env                    ← Secrets (jamais dans git)
├── .env.example            ← Template de configuration
├── .gitignore              ← Exclut data/, venv/, .env
├── .gitattributes          ← Normalise LF/CRLF
├── Dockerfile              ← Image Docker Python 3.12
├── docker-compose.yml      ← Services : app + ollama + indexer
├── requirements.txt        ← Dépendances Python
├── cours.md                ← Guide d'apprentissage de la stack
│
├── config/
│   └── settings.py         ← Chargement centralisé du .env
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py  ← Chunking hiérarchique + récursif
│   │   ├── vectorstore.py      ← Gestion ChromaDB
│   │   └── deduplication.py    ← Anti-doublons (seuil 0.92)
│   ├── llm/
│   │   └── llm_factory.py      ← Factory OpenAI / Ollama
│   ├── retrieval/
│   │   └── rag_chain.py        ← Pipeline RAG conversationnel
│   └── interface/
│       └── app.py              ← Interface Streamlit
│
├── scripts/
│   ├── build_index.py          ← Indexation initiale
│   ├── deduplicate_chroma.py   ← Nettoyage doublons
│   └── evaluate_rag.py         ← Évaluation qualité
│
├── tests/
│   ├── test_pipeline.py        ← Tests intégration pipeline
│   └── test_quality_tools.py   ← Tests déduplication et modes
│
└── data/                       ← Ignoré par git
    ├── docs/                   ← Documents source (PDFs...)
    ├── chroma_db/              ← Base vectorielle (générée)
    ├── embedding_cache/        ← Modèle HuggingFace (cache)
    ├── feedback/               ← Feedbacks utilisateurs (JSON)
    └── eval/                   ← Rapports d'évaluation
```

---

## 10. Résolution de problèmes

### `ollama` non reconnu dans PowerShell

```powershell
# Ajouter Ollama au PATH de la session
$env:PATH = "$env:LOCALAPPDATA\Programs\Ollama;" + $env:PATH
ollama --version

# Pour le rendre permanent : Paramètres Windows → Variables d'environnement
# Ajouter C:\Users\<user>\AppData\Local\Programs\Ollama à la variable PATH
```

### `git` non reconnu dans PowerShell

```powershell
$env:PATH = "C:\Program Files\Git\cmd;" + $env:PATH
git --version
```

### Erreur `Base ChromaDB introuvable`

```powershell
# La base n'a pas encore été construite — lancer l'indexation d'abord
python scripts/build_index.py
```

### Ollama ne répond pas (erreur de connexion)

```powershell
# Vérifier que le serveur tourne
curl http://localhost:11434/api/tags

# Si pas de réponse, démarrer le serveur
ollama serve

# Ou vérifier le processus
Get-Process ollama -ErrorAction SilentlyContinue
```

### Modèle Llama 3 non trouvé

```powershell
# Lister les modèles disponibles
ollama list

# Télécharger Llama 3
ollama pull llama3   # ~4.7 GB

# Tester
ollama run llama3 "test"
```

### Mémoire insuffisante pour Llama 3

```powershell
# Utiliser llama3:8b (plus léger) au lieu de llama3 (70B par défaut)
ollama pull llama3:8b

# Mettre à jour .env
# LLM_MODEL=llama3:8b
```

### `ModuleNotFoundError` au lancement

```powershell
# Vérifier que le venv est activé
venv\Scripts\activate
python -c "import langchain; print('OK')"

# Réinstaller si besoin
pip install -r requirements.txt
```

### Qualité des réponses insuffisante

```powershell
# 1. Vérifier le nombre de chunks indexés
python -c "
import sys; sys.path.insert(0,'.')
from src.ingestion.vectorstore import VectorStoreManager
vs = VectorStoreManager()
vs.load()
print('Chunks:', vs._vectorstore._collection.count())
"

# 2. Augmenter le nombre de chunks retournés
# Dans .env : RETRIEVER_K=6  (défaut : 4)

# 3. Ajouter plus de documents dans data/docs/ et réindexer
```

---

## Versions vérifiées

| Composant | Version |
|-----------|---------|
| Python | 3.12.9 |
| langchain | 0.2.16 |
| chromadb | 0.5.3 |
| sentence-transformers | 3.0.1 |
| streamlit | 1.37.0 |
| Ollama | 0.23.3 |
| Docker | 29.3.0 |

---

## Prochaines étapes

- **Phase 2** : Agents spécialisés CrewAI (Documentation, Réseau, Sécurité, Rapport)
- **Phase 3** : Audit automatisé (Nmap, OWASP ZAP, MobSF) + rapports automatiques
