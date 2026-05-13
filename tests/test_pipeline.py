# ── tests/test_pipeline.py ───────────────────────────────────────────────────
# Tests de validation du pipeline RAG.
# Utilisation :
#   pytest tests/test_pipeline.py -v
#   pytest tests/test_pipeline.py -v -k "test_chunking"

import sys
from pathlib import Path
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.vectorstore import VectorStoreManager, get_embedding_model
from config.settings import settings


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs():
    """Documents de test représentatifs du corpus cybersécurité."""
    return [
        Document(
            page_content=(
                "Wazuh est une plateforme open source de sécurité qui offre "
                "des capacités SIEM et XDR. Il permet la détection des menaces, "
                "la surveillance de l'intégrité des fichiers et la réponse aux incidents. "
                "L'agent Wazuh s'installe sur les endpoints Linux et Windows."
            ),
            metadata={"source": "wazuh_guide.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "Pour configurer un VPN OpenVPN sur pfSense, accéder à VPN > OpenVPN. "
                "Créer un nouveau serveur avec le protocole UDP sur le port 1194. "
                "Générer les certificats PKI via le menu System > Cert. Manager. "
                "Le tunnel utilise le chiffrement AES-256-GCM."
            ),
            metadata={"source": "pfsense_vpn.pdf", "page": 12},
        ),
        Document(
            page_content=(
                "Zabbix permet la supervision des infrastructures réseau. "
                "La configuration d'un host se fait via Configuration > Hosts. "
                "Les templates prédéfinis incluent Linux by Zabbix agent, "
                "Windows by Zabbix agent et Network devices by SNMP."
            ),
            metadata={"source": "zabbix_admin.pdf", "page": 5},
        ),
    ]


# ── Tests DocumentLoader ──────────────────────────────────────────────────────

class TestDocumentLoader:

    def test_split_produces_chunks(self, sample_docs):
        """Le splitter doit produire des chunks depuis les documents."""
        loader = DocumentLoader()
        chunks = loader.split(sample_docs)
        assert len(chunks) > 0, "Aucun chunk produit"

    def test_chunks_preserve_metadata(self, sample_docs):
        """Chaque chunk doit conserver les métadonnées du document source."""
        loader = DocumentLoader()
        chunks = loader.split(sample_docs)
        for chunk in chunks:
            assert "source" in chunk.metadata, "Métadonnée 'source' manquante"

    def test_chunk_size_respected(self, sample_docs):
        """Les chunks ne doivent pas dépasser CHUNK_SIZE + overlap."""
        loader = DocumentLoader()
        chunks = loader.split(sample_docs)
        max_allowed = settings.CHUNK_SIZE + settings.CHUNK_OVERLAP + 50
        for chunk in chunks:
            assert len(chunk.page_content) <= max_allowed, (
                f"Chunk trop grand : {len(chunk.page_content)} chars"
            )

    def test_load_directory_missing_raises(self):
        """Un dossier inexistant doit lever une FileNotFoundError."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory("/chemin/inexistant/xyz")

    def test_load_txt_file(self, tmp_path):
        """Vérifier le chargement d'un fichier .txt simple."""
        doc_file = tmp_path / "test.txt"
        doc_file.write_text(
            "Nmap est un scanner réseau open source.\n"
            "Il permet de découvrir les hôtes et services sur un réseau."
        )
        loader = DocumentLoader()
        docs = loader.load_file(str(doc_file))
        assert len(docs) > 0
        assert "Nmap" in docs[0].page_content

    def test_unsupported_format_returns_empty(self, tmp_path):
        """Un format non supporté doit retourner une liste vide sans erreur."""
        unsupported = tmp_path / "archive.zip"
        unsupported.write_bytes(b"fake zip content")
        loader = DocumentLoader()
        docs = loader.load_file(str(unsupported))
        assert docs == []


# ── Tests VectorStoreManager ──────────────────────────────────────────────────

class TestVectorStoreManager:

    def test_build_and_retrieve(self, sample_docs, tmp_path):
        """
        Test d'intégration complet : build → retriever → recherche.
        Utilise un dossier temporaire pour ne pas polluer la vraie base.
        """
        # Patcher les settings pour utiliser un répertoire temporaire
        original_dir = settings.CHROMA_PERSIST_DIR
        original_collection = settings.CHROMA_COLLECTION
        settings.CHROMA_PERSIST_DIR = str(tmp_path / "test_chroma")
        settings.CHROMA_COLLECTION = "test_collection"

        try:
            loader = DocumentLoader()
            chunks = loader.split(sample_docs)

            vs_manager = VectorStoreManager()
            vs_manager.build(chunks)

            # Test de recherche sémantique
            retriever = vs_manager.get_retriever(k=2)
            results = retriever.invoke("Comment configurer Wazuh ?")

            assert len(results) > 0, "Aucun résultat retourné"
            assert len(results) <= 2, "Plus de K résultats retournés"

            # Le document Wazuh doit être dans les résultats
            all_content = " ".join([r.page_content for r in results])
            assert "Wazuh" in all_content, "Document Wazuh non retrouvé"

        finally:
            settings.CHROMA_PERSIST_DIR = original_dir
            settings.CHROMA_COLLECTION = original_collection

    def test_load_nonexistent_raises(self, tmp_path):
        """Charger une base inexistante doit lever une FileNotFoundError."""
        settings.CHROMA_PERSIST_DIR = str(tmp_path / "inexistant")
        vs_manager = VectorStoreManager()
        with pytest.raises(FileNotFoundError):
            vs_manager.load()


# ── Test de cohérence du prompt RAG ──────────────────────────────────────────

class TestRAGPrompt:

    def test_prompt_contains_required_variables(self):
        """Le prompt doit contenir les 3 variables requises par LangChain."""
        from src.retrieval.rag_chain import RAG_PROMPT_TEMPLATE
        assert "{context}" in RAG_PROMPT_TEMPLATE
        assert "{question}" in RAG_PROMPT_TEMPLATE
        assert "{chat_history}" in RAG_PROMPT_TEMPLATE
