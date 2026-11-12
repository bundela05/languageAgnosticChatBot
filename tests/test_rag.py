import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
import chromadb
import pytest
from langchain_core.documents import Document

from app.rag.loader import CampusDocumentLoader
from app.rag.retriever import CampusRetriever
from app.rag.splitter import CampusDocumentSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_notices_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory populated with sample campus circulars."""
    notices_dir = tmp_path / "official_notices"
    notices_dir.mkdir(parents=True, exist_ok=True)

    # Sample Exam & Project Schedule Notice (matching BTCS706N context)
    exam_notice = notices_dir / "Notice_Project_BTCS706N_Schedule.txt"
    exam_notice.write_text(
        "Shri Vaishnav Institute of Information Technology\n"
        "NOTICE: Project (BTCS706N/BTIT706N) Academic Schedule\n"
        "Synopsis Submission: 1-Sep-2026 to 5-Sep-2026.\n"
        "Review-I by Guide: 15-Sep-2026 to 20-Sep-2026.\n"
        "Project External Examination: 23-Nov-2026 to 1-Dec-2026.\n"
        "Mandatory Submission of Two Research Papers with similarity < 15%.",
        encoding="utf-8"
    )

    # Sample Fee Notice
    fee_notice = notices_dir / "Semester_Fee_Notice_2026.txt"
    fee_notice.write_text(
        "Accounts & Student Finance Circular\n"
        "Last date to deposit B.Tech 7th Semester tuition fees without late fine is 10-Aug-2026.\n"
        "Late fee penalty of Rs. 500 applies after the deadline.",
        encoding="utf-8"
    )

    return notices_dir


@pytest.fixture
def mock_embedding_service():
    """
    Provides deterministic 4-dimensional unit vectors based on domain keywords
    to test cosine similarity and thresholding rapidly without downloading weights.
    """
    service = MagicMock()

    def fake_embed(text: str):
        t = text.lower()
        if any(w in t for w in ["project", "btcs706n", "synopsis", "paper", "exam", "schedule"]):
            return [0.95, 0.05, 0.0, 0.0]
        if any(w in t for w in ["fee", "tuition", "fine", "accounts", "deposit"]):
            return [0.0, 0.95, 0.05, 0.0]
        # Unrelated topic vector (orthogonal)
        return [0.0, 0.0, 0.0, 1.0]

    service.embed_query.side_effect = fake_embed
    service.embed_documents.side_effect = lambda texts: [fake_embed(txt) for txt in texts]
    return service


@pytest.fixture
def isolated_retriever(temp_notices_dir, mock_embedding_service):
    """Creates a CampusRetriever backed by an in-memory EphemeralClient."""
    ephemeral_client = chromadb.EphemeralClient()
    collection_name = f"test_campus_{uuid.uuid4().hex[:8]}"

    with patch("app.rag.retriever.VectorStoreManager.get_collection") as mock_get_col, \
         patch("app.rag.retriever.MultilingualEmbeddingService.get_instance", return_value=mock_embedding_service):
        
        mock_get_col.return_value = ephemeral_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        retriever = CampusRetriever(
            collection_name=collection_name,
            top_k=3,
            similarity_threshold=0.50
        )
        retriever.loader = CampusDocumentLoader(notices_dir=str(temp_notices_dir))
        yield retriever


# ---------------------------------------------------------------------------
# 1. Document Loader & Category Inference Tests
# ---------------------------------------------------------------------------

class TestCampusDocumentLoader:

    def test_load_all_notices_reads_directory(self, temp_notices_dir):
        loader = CampusDocumentLoader(notices_dir=str(temp_notices_dir))
        docs = loader.load_all_notices()

        assert len(docs) == 2
        sources = {d.metadata["source"] for d in docs}
        assert "Notice_Project_BTCS706N_Schedule.txt" in sources
        assert "Semester_Fee_Notice_2026.txt" in sources

    @pytest.mark.parametrize("filename,expected_category", [
        ("Semester_Tuition_Fee_Circular.pdf", "fees"),
        ("End_Sem_Exam_Time_Table.pdf", "exams"),
        ("Medhavi_Scholarship_Guidelines.pdf", "scholarships"),
        ("BTech_Counseling_Admission_2026.pdf", "admissions"),
        ("Annual_Holiday_Calendar.pdf", "campus_life"),
        ("General_Notice_395.pdf", "general"),
    ])
    def test_infer_category_from_filename(self, filename, expected_category):
        assert CampusDocumentLoader._infer_category(filename) == expected_category


# ---------------------------------------------------------------------------
# 2. Text Splitter & Table Preservation Tests
# ---------------------------------------------------------------------------

class TestCampusDocumentSplitter:

    def test_split_documents_attaches_chunk_metadata(self):
        splitter = CampusDocumentSplitter(chunk_size=120, chunk_overlap=20)
        long_text = (
            "Synopsis Submission is scheduled from 1-Sep-2026 to 5-Sep-2026.\n\n"
            "Review-I by Project Guide in hard copy and soft copy will take place from 15-Sep-2026 to 20-Sep-2026.\n\n"
            "Both research papers must have a similarity score below 15 percent on Turnitin or DrillBit."
        )
        raw_doc = Document(
            page_content=long_text,
            metadata={"source": "BTCS706N.pdf", "page": 1, "category": "exams"}
        )

        chunks = splitter.split_documents([raw_doc])

        assert len(chunks) >= 2
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "BTCS706N.pdf"
            assert chunk.metadata["chunk_id"] == idx
            assert chunk.metadata["char_length"] == len(chunk.page_content)
            assert len(chunk.page_content) >= 20

    def test_split_ignores_ultra_short_fragments(self):
        splitter = CampusDocumentSplitter(chunk_size=500, chunk_overlap=50)
        short_doc = Document(page_content="Page 1", metadata={"source": "empty.pdf"})
        chunks = splitter.split_documents([short_doc])
        assert chunks == []

    def test_split_preserves_table_blocks(self):
        splitter = CampusDocumentSplitter(chunk_size=250, chunk_overlap=30)
        table_doc = Document(
            page_content=(
                "Academic Schedule Overview:\n"
                "[TABLE DATA]\n"
                "S.No | Particular | Duration\n"
                "1 | Synopsis Submission | 1-Sep-2026 to 5-Sep-2026\n"
                "2 | External Exam | 23-Nov-2026 to 1-Dec-2026\n"
                "[/TABLE DATA]\n"
                "All students must adhere strictly to these dates."
            ),
            metadata={"source": "Schedule.pdf", "page": 1}
        )

        chunks = splitter.split_documents([table_doc])
        assert len(chunks) >= 1
        assert any("Synopsis Submission | 1-Sep-2026" in c.page_content for c in chunks)


# ---------------------------------------------------------------------------
# 3. Vector Indexing & Retrieval Tests (SRS T-03)
# ---------------------------------------------------------------------------

class TestCampusRetriever:

    def test_index_all_and_retrieve_relevant_notice(self, isolated_retriever):
        index_summary = isolated_retriever.index_all_documents()
        assert index_summary["status"] == "complete"
        assert index_summary["total_chunks"] >= 2

        # Query about project schedule
        citations = isolated_retriever.retrieve("When is BTCS706N synopsis submission?")
        assert len(citations) >= 1
        assert citations[0].document_title == "Notice_Project_BTCS706N_Schedule.txt"
        assert citations[0].relevance_score >= 0.50
        assert "1-Sep-2026" in citations[0].chunk_text

    def test_retrieve_filters_out_low_similarity_queries(self, isolated_retriever):
        isolated_retriever.index_all_documents()

        # Orthogonal out-of-scope query should score ~0.0 similarity and be filtered out
        citations = isolated_retriever.retrieve("Who won the cricket world cup?")
        assert citations == []

    def test_delete_document_records_purges_vectors(self, isolated_retriever):
        isolated_retriever.index_all_documents()

        # Verify fee notice is retrievable initially
        initial_hits = isolated_retriever.retrieve("What is the late fee fine?")
        assert len(initial_hits) >= 1

        # Delete fee notice records
        isolated_retriever.delete_document_records("Semester_Fee_Notice_2026.txt")

        # Verify fee notice is no longer returned
        post_delete_hits = isolated_retriever.retrieve("What is the late fee fine?")
        assert post_delete_hits == []
