import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.database.database import get_db_connection, log_audit_event
from app.models.schemas import DocumentUploadResponse
from app.rag.retriever import get_campus_retriever

router = APIRouter(prefix="/documents", tags=["Campus Documents & Knowledge Base"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./documents/official_notices"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_campus_notice(file: UploadFile = File(...)):
    """
    Uploads an official campus circular or notice (PDF/TXT),
    stores it in the documents repository, and indexes it into ChromaDB.
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Only PDF and TXT circulars are accepted."
        )

    # Sanitize and prepare destination path
    safe_filename = Path(file.filename).name
    destination_path = UPLOAD_DIR / safe_filename

    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save notice file to disk: {str(e)}"
        )

    # Index into ChromaDB via RAG pipeline
    try:
        retriever = get_campus_retriever()
        index_result = retriever.index_document(destination_path)

        # Track document in SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        doc_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT OR REPLACE INTO document_records (document_id, filename, category, total_pages, chunks_count, uploaded_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            doc_id,
            safe_filename,
            retriever.loader._infer_category(safe_filename),
            index_result.get("pages", 1),
            index_result.get("chunks", 0)
        ))
        conn.commit()
        conn.close()

        log_audit_event(
            action="DOCUMENT_UPLOAD",
            details=f"Uploaded and indexed notice '{safe_filename}' ({index_result.get('chunks', 0)} chunks)."
        )

        return DocumentUploadResponse(
            filename=safe_filename,
            total_pages=index_result.get("pages", 1),
            chunks_created=index_result.get("chunks", 0),
            status="indexed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing document into vector database: {str(e)}"
        )


@router.post("/reindex-all", response_model=Dict[str, Any])
async def reindex_all_notices():
    """
    Scans the official notices repository and re-indexes all PDF/TXT files into ChromaDB.
    Useful after manual document placement or system initialization.
    """
    try:
        retriever = get_campus_retriever()
        result = retriever.index_all_documents()
        
        log_audit_event(
            action="REINDEX_ALL",
            details=f"Reindexed entire notice repository: {result.get('total_documents', 0)} documents, {result.get('total_chunks', 0)} chunks."
        )
        return {
            "status": "success",
            "message": "All campus notices reindexed successfully.",
            "details": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex campus notices: {str(e)}"
        )


@router.get("/list", response_model=List[Dict[str, Any]])
async def list_indexed_documents():
    """
    Lists all campus notices tracked by the system along with page count and chunk statistics.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT document_id, filename, category, total_pages, chunks_count, uploaded_at
        FROM document_records
        ORDER BY uploaded_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "document_id": r["document_id"],
            "filename": r["filename"],
            "category": r["category"],
            "total_pages": r["total_pages"],
            "chunks_count": r["chunks_count"],
            "uploaded_at": str(r["uploaded_at"])
        }
        for r in rows
    ]


@router.delete("/{filename}", response_model=Dict[str, str])
async def delete_notice(filename: str):
    """
    Deletes a notice from disk, purges its vector embeddings from ChromaDB,
    and removes tracking records.
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notice '{filename}' not found on server."
        )

    try:
        # 1. Delete from ChromaDB
        retriever = get_campus_retriever()
        retriever.delete_document_records(filename)

        # 2. Delete file from disk
        file_path.unlink()

        # 3. Remove from SQLite tracking
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_records WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()

        log_audit_event(
            action="DOCUMENT_DELETE",
            details=f"Deleted notice '{filename}' and purged associated vector embeddings."
        )

        return {"status": "success", "message": f"Notice '{filename}' removed and purged."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete notice: {str(e)}"
        )
