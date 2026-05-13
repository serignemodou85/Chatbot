from langchain.schema import Document

from src.ingestion.deduplication import deduplicate_chunks
from src.retrieval.rag_chain import RAGChain


def test_deduplicate_chunks_removes_near_duplicates():
    chunks = [
        Document(page_content="Wazuh detecte les menaces sur les endpoints Linux."),
        Document(page_content="Wazuh detecte les menaces sur les endpoints Linux. "),
        Document(page_content="Zabbix supervise les infrastructures reseau."),
    ]

    deduped = deduplicate_chunks(chunks, similarity_threshold=0.98)
    assert len(deduped) == 2


def test_user_mode_instruction_mapping():
    student = RAGChain._build_mode_instruction("🎓 Étudiant")
    pro = RAGChain._build_mode_instruction("🔒 Pro cybersécurité")

    assert "debutant" in student.lower() or "débutant" in student.lower()
    assert "expert" in pro.lower()
