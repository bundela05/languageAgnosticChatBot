import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from app.models.schemas import MessageRecord

load_dotenv()

# Storage paths
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./campus_chatbot.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "campus_documents")

# Ensure required directory structures exist
os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. SQLite Relational Storage (Conversations, Messages, Audit Logs)
# ---------------------------------------------------------------------------

def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with row-factory enabled."""
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite_db() -> None:
    """Initializes tables for conversations, messages, documents, and logs."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            preferred_language TEXT DEFAULT 'auto',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            sender TEXT CHECK(sender IN ('user', 'assistant', 'system')) NOT NULL,
            text TEXT NOT NULL,
            language TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS document_records (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            total_pages INTEGER DEFAULT 1,
            chunks_count INTEGER DEFAULT 0,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id TEXT PRIMARY KEY,
            conversation_id TEXT,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def save_message(
    conversation_id: str,
    sender: str,
    text: str,
    language: str = "en"
) -> MessageRecord:
    """Inserts a message record and ensures the parent conversation exists."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Ensure parent conversation session exists
    cursor.execute("""
        INSERT OR IGNORE INTO conversations (conversation_id, preferred_language, created_at, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """, (conversation_id, language))

    cursor.execute("""
        UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?
    """, (conversation_id,))

    message_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO messages (message_id, conversation_id, sender, text, language, timestamp)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (message_id, conversation_id, sender, text, language))

    conn.commit()
    conn.close()

    return MessageRecord(
        message_id=message_id,
        conversation_id=conversation_id,
        sender=sender,  # type: ignore
        text=text,
        language=language,
        timestamp=datetime.utcnow()
    )


def fetch_conversation_history(conversation_id: str, limit: int = 20) -> List[MessageRecord]:
    """Retrieves chronological conversation message turns for context."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT message_id, conversation_id, sender, text, language, timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    """, (conversation_id, limit))

    rows = cursor.fetchall()
    conn.close()

    history: List[MessageRecord] = []
    for row in rows:
        history.append(
            MessageRecord(
                message_id=row["message_id"],
                conversation_id=row["conversation_id"],
                sender=row["sender"],
                text=row["text"],
                language=row["language"],
                timestamp=datetime.fromisoformat(str(row["timestamp"])) if isinstance(row["timestamp"], str) else row["timestamp"]
            )
        )
    return history


def log_audit_event(action: str, details: str, conversation_id: Optional[str] = None) -> None:
    """Logs system events and administrative queries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_logs (log_id, conversation_id, action, details, timestamp)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (str(uuid.uuid4()), conversation_id, action, details))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 2. Vector Store Setup (ChromaDB Persistent Client)
# ---------------------------------------------------------------------------

class VectorStoreManager:
    """Manages persistent ChromaDB client initialization and collection handles."""
    _client: Optional[chromadb.ClientAPI] = None

    @classmethod
    def get_client(cls) -> chromadb.ClientAPI:
        if cls._client is None:
            cls._client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
        return cls._client

    @classmethod
    def get_collection(cls, collection_name: str = COLLECTION_NAME):
        """Returns the target collection; creates it if absent."""
        client = cls.get_client()
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )


# Run schema initialization on module load
init_sqlite_db()
