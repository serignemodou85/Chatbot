# 🛡️ CyberSec RAG Chatbot — Phase 1

Chatbot IA spécialisé en cybersécurité, infrastructure réseau et administration
système. Basé sur une architecture RAG (Retrieval-Augmented Generation) avec
LangChain, ChromaDB et Llama 3 (ou GPT-4o).


## Structure du projet

```
rag_chatbot/
├── data/
│   ├── docs/              ← Déposer vos documents ici (PDF, DOCX, TXT, MD)
│   └── chroma_db/         ← Base vectorielle (générée automatiquement)
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py   ← Chargement + chunking des documents
│   │   └── vectorstore.py       ← Gestion de ChromaDB
│   ├── llm/
│   │   └── llm_factory.py       ← Factory OpenAI / Ollama
│   ├── retrieval/
│   │   └── rag_chain.py         ← Pipeline RAG conversationnel
│   └── interface/
│       └── app.py               ← Interface Streamlit
├── scripts/
│   └── build_index.py           ← Script d'indexation initiale
├── tests/
│   └── test_pipeline.py         ← Tests unitaires
├── config/
│   └── settings.py              ← Configuration centralisée
├── .env.example                 ← Template de configuration
└── requirements.txt


## Installation

### 1. Cloner et installer les dépendances

```bash
git clone <votre-repo>
cd rag_chatbot
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Configurer l'environnement

```bash
cp .env.example .env
# Éditer .env selon votre choix de LLM
```

### 3. Installer le LLM

**Option A — Llama 3 local (gratuit, recommandé pour commencer)**
```bash
# Installer Ollama : https://ollama.ai
ollama pull llama3
# Vérifier : ollama run llama3 "bonjour"
```

**Option B — GPT-4o (OpenAI)**
```bash
# Dans .env :
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o
# OPENAI_API_KEY=sk-...


## Utilisation

### Étape 1 — Préparer les documents

Déposer vos fichiers dans `data/docs/` :
- Documentation Wazuh (PDF)
- Guides pfSense / OpenVPN
- Docs Zabbix
- Procédures admin Linux/Windows
- Tout fichier PDF, DOCX, TXT ou Markdown

### Étape 2 — Indexer les documents (une seule fois)

```bash
python scripts/build_index.py
```

Options :
```bash
python scripts/build_index.py --docs-dir /chemin/perso    # Autre dossier
python scripts/build_index.py --reset                     # Réindexer tout
```

### Étape 3 — Lancer le chatbot

```bash
streamlit run src/interface/app.py
```

Ouvrir http://localhost:8501 dans le navigateur.


## Tests

```bash
pip install pytest
pytest tests/ -v
```

### Déduplication des chunks (anti-doublons)

```bash
python scripts/deduplicate_chroma.py --dry-run
python scripts/deduplicate_chroma.py
```

Options utiles :
```bash
python scripts/deduplicate_chroma.py --threshold 0.95
python scripts/deduplicate_chroma.py --collection cybersec_docs
```

### Évaluation de la pertinence RAG

```bash
python scripts/evaluate_rag.py
```

Le script produit un rapport JSON :
`data/eval/last_report.json`


## Configuration avancée (.env)

 Variable  Défaut  Description 

 `LLM_PROVIDER`  `ollama`  `ollama` ou `openai` 
 `LLM_MODEL`  `llama3`  Nom du modèle 
 `CHUNK_SIZE`  `512`  Taille des chunks (tokens) 
 `CHUNK_OVERLAP`  `64`  Chevauchement entre chunks 
 `RETRIEVER_K`  `4`  Chunks retournés par requête 
 `EMBEDDING_PROVIDER`  `huggingface`  `huggingface` ou `openai` 


# si t'as ces genres d'erreurs 
###  Solutions appliquées
- Migration vers Python 3.12 (version stable pour le machine learning)
- Correction des versions dans `requirements.txt` :
  - chromadb ajusté pour compatibilité avec LangChain
- Installation de Microsoft C++ Build Tools pour les dépendances compilées
- Reconstruction de l’environnement virtuel

## Prochaines étapes

- **Phase 2** : Agents spécialisés avec CrewAI (Agent Réseau, Agent Sécurité...)
- **Phase 3** : Audit automatisé (Nmap, OWASP ZAP, MobSF)



