# ── Dockerfile ────────────────────────────────────────────────────────────────
# Image de production pour le RAG Chatbot (Streamlit + LangChain + ChromaDB)
#
# Build :  docker build -t rag-chatbot .
# Run  :   docker compose up
#
# Architecture du cache Docker (couches dans l'ordre du moins au plus volatile) :
#   1. Image de base Python (change rarement)
#   2. Dépendances système
#   3. requirements.txt (change quand on ajoute une lib)
#   4. Code source (change souvent)

# ── Étape 1 : image de base ───────────────────────────────────────────────────
# python:3.11-slim = Python 3.11 sur Debian minimal (~130 MB vs ~900 MB pour python:3.11)
FROM python:3.11-slim

# Métadonnées de l'image
LABEL maintainer="tellofall@gmail.com"
LABEL description="RAG Chatbot - Cybersécurité & Infrastructure réseau"
LABEL version="1.0"

# ── Étape 2 : variables d'environnement du container ──────────────────────────
# PYTHONDONTWRITEBYTECODE : pas de fichiers .pyc dans l'image
# PYTHONUNBUFFERED : logs en temps réel (stdout/stderr non bufférisés)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Dossier de travail dans le container
    WORKDIR_PATH=/app

# ── Étape 3 : dépendances système ─────────────────────────────────────────────
# Nécessaires pour : chromadb (sqlite3), sentence-transformers (tokenizers), pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Étape 4 : répertoire de travail ───────────────────────────────────────────
WORKDIR /app

# ── Étape 5 : dépendances Python (couche cachée séparée du code) ──────────────
# Copier UNIQUEMENT requirements.txt d'abord.
# Si le code change mais pas les dépendances → Docker réutilise cette couche.
# C'est l'optimisation la plus importante du Dockerfile.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Étape 6 : code source ─────────────────────────────────────────────────────
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# ── Étape 7 : structure des dossiers de données ───────────────────────────────
# Ces dossiers seront montés via volumes en docker-compose.
# On les crée ici pour que l'app ne crash pas si les volumes ne sont pas montés.
RUN mkdir -p \
    data/docs \
    data/chroma_db \
    data/embedding_cache \
    data/feedback \
    data/eval \
    logs

# ── Étape 8 : utilisateur non-root (bonne pratique de sécurité) ──────────────
# Jamais lancer un container en root en production.
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# ── Étape 9 : port et démarrage ───────────────────────────────────────────────
# Port par défaut de Streamlit
EXPOSE 8501

# Healthcheck : vérifie que Streamlit répond toutes les 30 secondes
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Commande de démarrage
# --server.address=0.0.0.0 : écouter sur toutes les interfaces (pas juste localhost)
# --server.headless=true   : pas de popup navigateur automatique
CMD ["streamlit", "run", "src/interface/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
