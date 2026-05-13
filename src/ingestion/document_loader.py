# ── src/ingestion/document_loader.py ─────────────────────────────────────────
# Stratégie de chunking : Hierarchical + Recursive
#
# Pourquoi cette combinaison pour de la doc réseau/sécu ?
#
#   Les docs techniques (Wazuh, Zabbix, pfSense...) sont structurés en sections :
#   ## Installation -> ## Configuration -> ## Troubleshooting
#
#   Problème du chunking naïf (Fixed Size) :
#     Un chunk peut contenir la fin de "Installation" ET le début de
#     "Configuration" -> le LLM reçoit un contexte incohérent -> mauvaise réponse.
#
#   Solution Hierarchical + Recursive :
#     1. On coupe d'abord sur les titres Markdown (# ## ###)
#        -> chaque section reste entière dans son propre chunk parent
#     2. Si une section depasse CHUNK_SIZE, on la redécoupe avec
#        RecursiveCharacterTextSplitter (paragraphe -> phrase -> mot)
#     3. Chaque chunk hérite du titre de sa section en métadonnée
#        -> le LLM sait toujours "dans quelle section" il se trouve
#
# Formats supportés : PDF, DOCX, TXT, Markdown, HTML

import re
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from loguru import logger

from config.settings import settings


# Association extension -> classe de loader LangChain
LOADERS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
    ".html": UnstructuredHTMLLoader,
}

# Séparateurs hiérarchiques Markdown
MARKDOWN_HEADERS = [
    ("#",   "titre_h1"),
    ("##",  "titre_h2"),
    ("###", "titre_h3"),
]

# Séparateurs RecursiveCharacterTextSplitter pour docs techniques réseau.
# Ordre : section -> paragraphe -> ligne -> phrase -> mot
# On ajoute les blocs de code pour ne jamais couper dedans.
TECHNICAL_SEPARATORS = [
    "\n## ", "\n### ",   # Titres de section (fallback si pas Markdown)
    "\n\n",              # Paragraphes
    "\n",                # Lignes
    "```\n",             # Blocs de code
    ". ",                # Phrases
    ", ",                # Propositions
    " ",                 # Mots
    "",                  # Caractères (dernier recours)
]


class DocumentLoader:
    """
    Charge et découpe les documents avec une stratégie Hierarchical + Recursive.

    Pour les fichiers Markdown (.md) : MarkdownHeaderTextSplitter en premier,
    puis RecursiveCharacterTextSplitter si une section dépasse CHUNK_SIZE.

    Pour les autres formats (PDF, DOCX, TXT) : RecursiveCharacterTextSplitter
    avec séparateurs techniques enrichis.
    """

    def __init__(self,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None):
        self.chunk_size    = chunk_size    or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Splitter Markdown : coupe sur les titres # ## ###
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS,
            strip_headers=False,  # Garder le titre dans le texte du chunk
        )

        # Splitter de redécoupage pour sections trop grandes ou formats non-MD
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=TECHNICAL_SEPARATORS,
            length_function=len,
        )

    # ── Chargement ────────────────────────────────────────────────────────────

    def load_file(self, file_path: str) -> List[Document]:
        """Charge un fichier et retourne ses sections brutes."""
        ext = Path(file_path).suffix.lower()
        loader_class = LOADERS.get(ext)

        if not loader_class:
            logger.warning(f"Format non supporté, ignoré : {file_path}")
            return []

        try:
            docs = loader_class(file_path).load()
            for doc in docs:
                doc.metadata.setdefault("source", file_path)
                doc.metadata["filename"]  = Path(file_path).name
                doc.metadata["extension"] = ext
            logger.info(f"  Chargé : {Path(file_path).name} ({len(docs)} section(s))")
            return docs
        except Exception as e:
            logger.error(f"  Erreur chargement {file_path} : {e}")
            return []

    def load_directory(self, docs_dir: str) -> List[Document]:
        """Charge récursivement tous les fichiers d'un dossier."""
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Dossier introuvable : {docs_dir}")

        files = [f for f in docs_path.rglob("*") if f.is_file()]
        logger.info(f"Fichiers trouvés dans {docs_dir} : {len(files)}")

        all_docs = []
        for file_path in files:
            all_docs.extend(self.load_file(str(file_path)))

        logger.info(f"Total sections chargées : {len(all_docs)}")
        return all_docs

    # ── Chunking ──────────────────────────────────────────────────────────────

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Découpe intelligente selon le type de document.

        Markdown -> MarkdownHeaderTextSplitter + redécoupage si nécessaire
        Autres   -> RecursiveCharacterTextSplitter avec séparateurs techniques
        """
        all_chunks = []

        for doc in documents:
            ext = doc.metadata.get("extension", "")

            if ext == ".md":
                chunks = self._split_markdown(doc)
            else:
                chunks = self._split_recursive(doc)

            # Enrichir les métadonnées de chaque chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(chunks)
                chunk.metadata.setdefault("source", doc.metadata.get("source", ""))
                chunk.metadata["preview"] = chunk.page_content[:80].replace("\n", " ")

            all_chunks.extend(chunks)

        logger.info(f"Total chunks produits : {len(all_chunks)}")
        self._log_chunk_stats(all_chunks)
        return all_chunks

    def _split_markdown(self, doc: Document) -> List[Document]:
        """
        Stratégie Hierarchical pour les fichiers Markdown.

        Etape 1 : MarkdownHeaderTextSplitter découpe sur # ## ###
                  -> chaque section devient un Document avec son titre en métadonnée
        Etape 2 : Si une section dépasse chunk_size, RecursiveCharacterTextSplitter
                  la redécoupe en préservant les métadonnées du parent
        """
        md_chunks = self.md_splitter.split_text(doc.page_content)

        # Transférer les métadonnées source vers les chunks
        for chunk in md_chunks:
            chunk.metadata.update({
                k: v for k, v in doc.metadata.items()
                if k not in chunk.metadata
            })

        # Redécouper les chunks trop grands
        final_chunks = []
        for chunk in md_chunks:
            if len(chunk.page_content) > self.chunk_size:
                sub_chunks = self.recursive_splitter.split_documents([chunk])
                for sub in sub_chunks:
                    sub.metadata.update(chunk.metadata)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _split_recursive(self, doc: Document) -> List[Document]:
        """
        Stratégie Recursive pour PDF, DOCX, TXT.

        On pré-traite le texte pour normaliser les titres fréquents
        dans les docs Wazuh/Zabbix/pfSense afin qu'ils servent de points
        de coupure naturels.
        """
        text = self._normalize_technical_headers(doc.page_content)
        normalized_doc = Document(page_content=text, metadata=doc.metadata)
        return self.recursive_splitter.split_documents([normalized_doc])

    def _normalize_technical_headers(self, text: str) -> str:
        """
        Convertit les patterns de titres fréquents dans les docs réseau
        en séparateurs pour améliorer les points de coupure.

        Exemples :
          "1. Installation"     -> "\n## 1. Installation"
          "CONFIGURATION"       -> "\n## CONFIGURATION"
        """
        # Ligne numérotée type "1. Titre" ou "2.3 Titre"
        text = re.sub(
            r'(?m)^(\d+\.[\d\.]*\s+[A-Z][^\n]{3,60})$',
            r'\n## \1',
            text
        )
        # Titre en MAJUSCULES seul sur sa ligne
        text = re.sub(
            r'(?m)^([A-Z][A-Z\s]{3,50})$',
            r'\n## \1',
            text
        )
        return text

    # ── Méthode principale ────────────────────────────────────────────────────

    def load_and_split(self, docs_dir: str) -> List[Document]:
        """Méthode principale : charge + découpe en une seule étape."""
        docs = self.load_directory(docs_dir)
        return self.split(docs)

    # ── Stats de débogage ─────────────────────────────────────────────────────

    def _log_chunk_stats(self, chunks: List[Document]):
        """Affiche des statistiques sur les chunks produits."""
        if not chunks:
            return
        sizes = [len(c.page_content) for c in chunks]
        logger.info(
            f"Stats chunks — "
            f"min: {min(sizes)} chars | "
            f"max: {max(sizes)} chars | "
            f"moy: {sum(sizes)//len(sizes)} chars"
        )
        tiny = [c for c in chunks if len(c.page_content) < 50]
        if tiny:
            logger.warning(
                f"{len(tiny)} chunk(s) tres courts (<50 chars). "
                "Vérifier la qualité des documents source."
            )
