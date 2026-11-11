import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.database.database import init_sqlite_db, log_audit_event
from app.rag.retriever import get_campus_retriever

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DOCUMENTS_DIR = BASE_DIR / "documents" / "official_notices"

# Ensure essential paths exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Executes startup checks and cleanup routines.
    """
    # 1. Initialize SQLite database schema
    init_sqlite_db()

    # 2. Pre-index existing notices in documents/official_notices if present
    try:
        retriever = get_campus_retriever()
        indexed_files = list(DOCUMENTS_DIR.glob("*.pdf")) + list(DOCUMENTS_DIR.glob("*.txt"))
        if indexed_files:
            print(f"[Lifespan] Found {len(indexed_files)} campus document(s). Syncing index...")
            result = retriever.index_all_documents()
            print(f"[Lifespan] Index sync complete: {result.get('total_chunks', 0)} chunks active.")
        else:
            print("[Lifespan] Notice directory is ready. No initial documents found to index.")
    except Exception as e:
        print(f"[Lifespan] Warning during startup document indexing: {e}")

    log_audit_event(action="SYSTEM_STARTUP", details="FastAPI server initialized successfully.")

    yield

    # Teardown logic
    log_audit_event(action="SYSTEM_SHUTDOWN", details="FastAPI server shutting down gracefully.")


# Initialize FastAPI application
app = FastAPI(
    title="Language-Agnostic Campus Chatbot API",
    description=(
        "An AI-driven multilingual document-grounded conversational system "
        "for university circulars, academic notices, and student inquiries."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# Middleware: CORS Configuration
# ---------------------------------------------------------------------------
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5173",
    "*"  # Allows access from local network testing and student devices
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["System Health"], status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint verifying server operational status and storage links.
    """
    return {
        "status": "healthy",
        "service": "Language-Agnostic Campus Chatbot API",
        "version": "1.0.0",
        "notices_directory_exists": DOCUMENTS_DIR.exists(),
        "frontend_mounted": FRONTEND_DIR.exists()
    }


# ---------------------------------------------------------------------------
# Static Frontend Serving
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", tags=["Frontend"])
    async def serve_index():
        """Serves the main single-page campus chat interface."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "index.html not found. Place your frontend files in the /frontend directory."}
        )
