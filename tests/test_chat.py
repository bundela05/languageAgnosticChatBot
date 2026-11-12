import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import database as db_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_sqlite_db(tmp_path: Path, monkeypatch):
    """Redirects SQLite database operations to a temporary test database."""
    test_db_path = tmp_path / "test_campus_chatbot.db"
    monkeypatch.setattr(db_module, "SQLITE_DB_PATH", str(test_db_path))
    db_module.init_sqlite_db()
    return test_db_path


@pytest.fixture
def mock_workflow():
    """Mocks CampusChatbotWorkflow to return deterministic state outputs without external LLM calls."""
    workflow = MagicMock()

    def fake_run(query: str, conversation_id: str, preferred_language: str = "auto", history=None):
        is_hindi = preferred_language == "hi" or "तिथि" in query
        return {
            "query": query,
            "conversation_id": conversation_id,
            "detected_language": "hi" if is_hindi else "en",
            "intent": "campus_academics",
            "final_response": (
                "सिनॉप्सिस जमा करने की अंतिम तिथि 5-Sep-2026 है।"
                if is_hindi
                else "As per notice BTCS706N, Synopsis Submission is scheduled from 1-Sep-2026 to 5-Sep-2026."
            ),
            "sources": [
                {
                    "document_title": "Notice_Project_BTCS706N.pdf",
                    "page_number": 1,
                    "chunk_text": "Synopsis Submission: 1-Sep-2026 to 5-Sep-2026.",
                    "relevance_score": 0.91
                }
            ]
        }

    workflow.run.side_effect = fake_run
    return workflow


@pytest.fixture
def client(isolated_sqlite_db, mock_workflow, tmp_path: Path, monkeypatch):
    """Provides a configured FastAPI TestClient with mocked workflow and isolated directories."""
    test_upload_dir = tmp_path / "official_notices"
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.api.documents.UPLOAD_DIR", test_upload_dir)

    with patch("app.api.chat.get_chatbot_workflow", return_value=mock_workflow), \
         patch("app.main.get_campus_retriever") as mock_startup_retriever:
        mock_startup_retriever.return_value.index_all_documents.return_value = {"total_chunks": 0}
        with TestClient(app) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# 1. Health Check & Frontend Mount Tests
# ---------------------------------------------------------------------------

class TestSystemEndpoints:

    def test_health_check_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# 2. Chat Invocation, History & Validation Tests (SRS T-04, T-05, T-06)
# ---------------------------------------------------------------------------

class TestChatEndpoints:

    def test_post_chat_valid_query_creates_session_and_returns_citations(self, client):
        response = client.post(
            "/api/chat",
            json={
                "message": "When is the BTCS706N synopsis submission?",
                "preferred_language": "auto"
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert "conversation_id" in data and len(data["conversation_id"]) > 0
        assert "5-Sep-2026" in data["response"]
        assert data["detected_language"] == "en"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_title"] == "Notice_Project_BTCS706N.pdf"

    def test_post_chat_empty_or_whitespace_returns_400(self, client):
        # Blank string fails custom endpoint check (SRS T-06 / E-01)
        response = client.post(
            "/api/chat",
            json={"message": "    ", "preferred_language": "en"}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_post_chat_missing_message_field_returns_422(self, client):
        response = client.post("/api/chat", json={"preferred_language": "en"})
        assert response.status_code == 422

    def test_multi_turn_conversation_history_persistence(self, client, mock_workflow):
        # Turn 1: Start new conversation
        turn1 = client.post(
            "/api/chat",
            json={"message": "Tell me about BTCS706N project dates."}
        ).json()
        convo_id = turn1["conversation_id"]

        # Turn 2: Follow-up in the same conversation session
        turn2 = client.post(
            "/api/chat",
            json={
                "message": "What about research paper plagiarism rules?",
                "conversation_id": convo_id
            }
        )
        assert turn2.status_code == 200

        # Verify workflow received prior turns in `history`
        second_call_kwargs = mock_workflow.run.call_args.kwargs
        assert len(second_call_kwargs["history"]) == 2  # Turn 1 user + Turn 1 assistant

        # Verify GET /api/chat/history/{conversation_id} returns all 4 stored messages
        history_res = client.get(f"/api/chat/history/{convo_id}")
        assert history_res.status_code == 200
        history_data = history_res.json()
        assert history_data["conversation_id"] == convo_id
        assert history_data["total_messages"] == 4
        assert [m["sender"] for m in history_data["messages"]] == ["user", "assistant", "user", "assistant"]

    def test_stream_chat_sse_endpoint(self, client):
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "Synopsis submission date?", "preferred_language": "en"}
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]

            assert len(lines) >= 3
            assert lines[-1] == "data: [DONE]"

            # First SSE payload should be metadata
            first_event = json.loads(lines[0].replace("data: ", ""))
            assert first_event["type"] == "metadata"
            assert first_event["detected_language"] == "en"


# ---------------------------------------------------------------------------
# 3. Document Upload, List & Delete Endpoints Tests (SRS T-07)
# ---------------------------------------------------------------------------

class TestDocumentEndpoints:

    def test_upload_list_and_delete_notice_lifecycle(self, client):
        mock_retriever = MagicMock()
        mock_retriever.index_document.return_value = {
            "filename": "Circular_2026.txt",
            "pages": 1,
            "chunks": 3,
            "status": "indexed"
        }
        mock_retriever.loader._infer_category.return_value = "general"

        with patch("app.api.documents.get_campus_retriever", return_value=mock_retriever):
            # 1. Upload valid TXT notice
            file_content = b"Official Campus Circular: Classes commence from 27 July 2026."
            upload_res = client.post(
                "/api/documents/upload",
                files={"file": ("Circular_2026.txt", file_content, "text/plain")}
            )
            assert upload_res.status_code == 201
            upload_data = upload_res.json()
            assert upload_data["filename"] == "Circular_2026.txt"
            assert upload_data["chunks_created"] == 3

            # 2. Verify notice appears in GET /api/documents/list
            list_res = client.get("/api/documents/list")
            assert list_res.status_code == 200
            docs = list_res.json()
            assert len(docs) == 1
            assert docs[0]["filename"] == "Circular_2026.txt"

            # 3. Delete notice via DELETE /api/documents/{filename}
            del_res = client.delete("/api/documents/Circular_2026.txt")
            assert del_res.status_code == 200
            mock_retriever.delete_document_records.assert_called_once_with("Circular_2026.txt")

            # 4. Confirm list is empty after deletion
            list_after = client.get("/api/documents/list").json()
            assert list_after == []

    def test_upload_unsupported_file_extension_returns_400(self, client):
        res = client.post(
            "/api/documents/upload",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")}
        )
        assert res.status_code == 400
        assert "unsupported file format" in res.json()["detail"].lower()
