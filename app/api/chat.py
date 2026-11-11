import json
import uuid
from datetime import datetime
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.chatbot.graph import get_chatbot_workflow
from app.database.database import (
    fetch_conversation_history,
    log_audit_event,
    save_message
)
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    MessageRecord,
    SourceCitation
)

router = APIRouter(prefix="/chat", tags=["Conversational Chatbot"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def send_chat_message(payload: ChatRequest):
    """
    Standard synchronous chat invocation endpoint.
    Processes query through LangGraph workflow:
    Detection -> Intent -> Vector Retrieval -> Grounded Synthesis -> Translation.
    Persists turns into SQLite session memory.
    """
    cleaned_message = payload.message.strip()
    if not cleaned_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message query cannot be empty or blank."
        )

    # 1. Resolve or establish conversational session UUID
    conversation_id = payload.conversation_id or str(uuid.uuid4())

    # 2. Fetch past conversation turns for memory context
    history_records = fetch_conversation_history(conversation_id, limit=6)
    history_turns = [
        {"sender": rec.sender, "text": rec.text}
        for rec in history_records
    ]

    # 3. Persist incoming student query
    save_message(
        conversation_id=conversation_id,
        sender="user",
        text=cleaned_message,
        language=payload.preferred_language or "auto"
    )

    try:
        # 4. Invoke LangGraph StateGraph pipeline
        workflow = get_chatbot_workflow()
        final_state = workflow.run(
            query=cleaned_message,
            conversation_id=conversation_id,
            preferred_language=payload.preferred_language or "auto",
            history=history_turns
        )

        response_text = final_state.get("final_response", "")
        detected_lang = final_state.get("detected_language", "en")
        intent = final_state.get("intent", "campus_academics")
        raw_sources = final_state.get("sources", [])

        # Format citations
        citations = [
            SourceCitation(
                document_title=s.get("document_title", "Campus Notice"),
                page_number=s.get("page_number", 1),
                chunk_text=s.get("chunk_text", ""),
                relevance_score=s.get("relevance_score")
            )
            for s in raw_sources
        ]

        # 5. Persist assistant response
        save_message(
            conversation_id=conversation_id,
            sender="assistant",
            text=response_text,
            language=detected_lang
        )

        # 6. Audit logging
        log_audit_event(
            action="CHAT_QUERY",
            details=f"Lang: {detected_lang} | Intent: {intent} | Sources: {len(citations)}",
            conversation_id=conversation_id
        )

        return ChatResponse(
            conversation_id=conversation_id,
            response=response_text,
            detected_language=detected_lang,
            intent=intent,
            sources=citations,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        log_audit_event(
            action="CHAT_ERROR",
            details=f"Error executing chat pipeline: {str(e)}",
            conversation_id=conversation_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the campus response: {str(e)}"
        )


@router.post("/stream")
async def stream_chat_message(payload: ChatRequest):
    """
    Streaming chat endpoint returning Server-Sent Events (SSE).
    Provides real-time token delivery to the web client.
    """
    cleaned_message = payload.message.strip()
    if not cleaned_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message query cannot be empty or blank."
        )

    conversation_id = payload.conversation_id or str(uuid.uuid4())

    history_records = fetch_conversation_history(conversation_id, limit=6)
    history_turns = [
        {"sender": rec.sender, "text": rec.text}
        for rec in history_records
    ]

    # Save user message
    save_message(
        conversation_id=conversation_id,
        sender="user",
        text=cleaned_message,
        language=payload.preferred_language or "auto"
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        workflow = get_chatbot_workflow()
        
        # Execute workflow graph
        state = workflow.run(
            query=cleaned_message,
            conversation_id=conversation_id,
            preferred_language=payload.preferred_language or "auto",
            history=history_turns
        )

        response_text = state.get("final_response", "")
        detected_lang = state.get("detected_language", "en")
        intent = state.get("intent", "campus_academics")
        sources = state.get("sources", [])

        # Persist response in SQLite
        save_message(
            conversation_id=conversation_id,
            sender="assistant",
            text=response_text,
            language=detected_lang
        )

        # Stream chunk payload
        meta_event = {
            "type": "metadata",
            "conversation_id": conversation_id,
            "detected_language": detected_lang,
            "intent": intent,
            "sources": sources
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        # Stream response text chunks
        chunk_size = 20
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            data_event = {
                "type": "content",
                "delta": chunk
            }
            yield f"data: {json.dumps(data_event)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.get("/history/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_chat_history(conversation_id: str, limit: int = Query(default=30, ge=1, le=100)):
    """
    Retrieves full message turn history for a given conversation session.
    """
    records = fetch_conversation_history(conversation_id=conversation_id, limit=limit)
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        total_messages=len(records),
        messages=records
    )
