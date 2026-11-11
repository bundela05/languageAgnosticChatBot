from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. API Request & Response Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming user chat query payload."""
    message: str = Field(
        ...,
        min_length=1,
        description="The student or user message/query.",
        examples=["When is the last date to submit the exam fees?"]
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Unique UUID for conversational session tracking. If None, a new session is generated."
    )
    preferred_language: Optional[str] = Field(
        default="auto",
        description="Optional language preference (e.g., 'en', 'hi', 'auto'). Defaults to auto-detection."
    )


class SourceCitation(BaseModel):
    """Metadata regarding retrieved campus circulars/notices supporting the response."""
    document_title: str = Field(..., description="Original filename or title of the source notice.")
    page_number: Optional[int] = Field(default=None, description="Page index where relevant text was located.")
    chunk_text: str = Field(..., description="Extracted excerpt matching the query context.")
    relevance_score: Optional[float] = Field(default=None, description="Similarity score from vector search.")


class ChatResponse(BaseModel):
    """Outgoing response payload returned to the student interface."""
    conversation_id: str = Field(..., description="Session identifier for tracking turns.")
    response: str = Field(..., description="Generated answer grounded in campus circulars.")
    detected_language: str = Field(..., description="Detected query language code (e.g., 'en', 'hi', 'hinglish').")
    intent: str = Field(..., description="Classified intent (e.g., 'campus_query', 'general_greeting', 'out_of_scope').")
    sources: List[SourceCitation] = Field(default_factory=list, description="Verified institutional document citations.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC response generation timestamp.")


# ---------------------------------------------------------------------------
# 2. Document & Ingestion Schemas
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Metadata attributes associated with stored document chunks in ChromaDB/PostgreSQL."""
    source: str = Field(..., description="File name or URL of the ingested campus document.")
    page: Optional[int] = Field(default=1, description="Page number of the document.")
    category: Optional[str] = Field(
        default="general",
        description="Administrative division: 'fees', 'exams', 'scholarships', 'admissions', 'general'."
    )
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp.")


class DocumentUploadResponse(BaseModel):
    """Response returned upon processing and indexing an official PDF circular."""
    filename: str = Field(..., description="Uploaded document filename.")
    total_pages: int = Field(..., description="Number of parsed pages.")
    chunks_created: int = Field(..., description="Number of vector chunks generated and indexed.")
    status: str = Field(default="indexed", description="Processing status.")
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 3. Database Record Schemas (SQLite / Logging)
# ---------------------------------------------------------------------------

class MessageRecord(BaseModel):
    """Model representing an individual stored message in a conversation."""
    message_id: str
    conversation_id: str
    sender: Literal["user", "assistant", "system"]
    text: str
    language: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistoryResponse(BaseModel):
    """Payload returning past message thread history."""
    conversation_id: str
    total_messages: int
    messages: List[MessageRecord]


# ---------------------------------------------------------------------------
# 4. LangGraph AgentState Schema
# ---------------------------------------------------------------------------

def append_messages(existing: List[Dict[str, str]], new_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Reducer function to append turns to the message history."""
    return existing + new_messages


class AgentState(TypedDict):
    """
    The internal state graph dictionary passed across LangGraph nodes.
    Flow: input_query -> language_detector -> intent_evaluator -> retriever -> synthesizer -> translator
    """
    # Core query inputs
    query: str
    conversation_id: str
    preferred_language: str
    
    # NLP & Language Pipeline State
    detected_language: str
    is_multilingual: bool
    translated_query_en: str
    intent: str
    
    # Knowledge Retrieval & Context
    retrieved_documents: List[Dict[str, Any]]
    context_found: bool
    
    # Generation & Delivery State
    draft_response: str
    final_response: str
    sources: List[Dict[str, Any]]
    
    # Historical turn memory
    history: Annotated[List[Dict[str, str]], append_messages]
