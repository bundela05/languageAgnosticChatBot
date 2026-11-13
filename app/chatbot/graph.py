import os
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from app.chatbot.intent import get_intent_classifier
from app.chatbot.language import get_language_service
from app.chatbot.prompts import (
    CAMPUS_RAG_PROMPT,
    FALLBACK_MESSAGES,
    SYSTEM_CAMPUS_GROUNDING_DIRECTIVE
)
from app.models.schemas import AgentState
from app.rag.retriever import get_campus_retriever

load_dotenv(override=True)


class CampusChatbotWorkflow:
    """
    LangGraph StateGraph workflow orchestrating multilingual query processing,
    intent classification, dense document retrieval, answer synthesis, and localization.
    """

    def __init__(self, llm: BaseChatModel = None):
        api_key = os.getenv("GROQ_API_KEY", "")
        model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        self.llm = llm or ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.1
        )
        self.lang_service = get_language_service()
        self.intent_classifier = get_intent_classifier()
        self.retriever = get_campus_retriever()

        # Build and compile graph
        self.graph = self._build_graph()

    # -----------------------------------------------------------------------
    # Node 1: Language Detection
    # -----------------------------------------------------------------------
    def detect_language_node(self, state: AgentState) -> Dict[str, Any]:
        """Detects incoming query language and script."""
        query = state.get("query", "")
        preferred = state.get("preferred_language", "auto")

        lang_code, _ = self.lang_service.detect_language(query, preferred_language=preferred)
        is_multi = lang_code in ("hi", "hinglish")

        return {
            "detected_language": lang_code,
            "is_multilingual": is_multi
        }

    # -----------------------------------------------------------------------
    # Node 2: Intent Classification & Routing
    # -----------------------------------------------------------------------
    def classify_intent_node(self, state: AgentState) -> Dict[str, Any]:
        """Categorizes query to determine if vector retrieval is necessary."""
        query = state.get("query", "")
        intent_data = self.intent_classifier.classify_intent(query)
        return {
            "intent": intent_data.get("intent", "campus_academics")
        }

    # -----------------------------------------------------------------------
    # Node 3: Cross-Lingual Knowledge Retrieval
    # -----------------------------------------------------------------------
    def retrieve_context_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Normalizes query into canonical English (if submitted in Hindi/Hinglish)
        and retrieves relevant notice excerpts from ChromaDB.
        """
        query = state.get("query", "")
        detected_lang = state.get("detected_language", "en")

        # 1. Normalize query for retrieval over formal English notices
        search_query = self.lang_service.normalize_query_for_retrieval(query, detected_lang)

        # 2. Retrieve top-k relevant document excerpts
        citations = self.retriever.retrieve(query=search_query, top_k=4)

        retrieved_docs: List[Dict[str, Any]] = [
            {
                "document_title": c.document_title,
                "page_number": c.page_number,
                "chunk_text": c.chunk_text,
                "relevance_score": c.relevance_score
            }
            for c in citations
        ]

        return {
            "translated_query_en": search_query,
            "retrieved_documents": retrieved_docs,
            "sources": retrieved_docs,
            "context_found": len(retrieved_docs) > 0
        }

    # -----------------------------------------------------------------------
    # Node 4: Answer Synthesis (Factual Grounding)
    # -----------------------------------------------------------------------
    def synthesize_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesizes the response strictly grounded in retrieved notices.
        If no context exists or query is conversational, applies appropriate handlers.
        """
        intent = state.get("intent", "")
        detected_lang = state.get("detected_language", "en")
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_documents", [])
        context_found = state.get("context_found", False)

        # Handle greetings directly without running RAG chain
        if intent == "general_greeting":
            greetings_map = {
                "en": "Hello! I am your Campus Virtual Assistant. How can I assist you with exam dates, fee notices, scholarships, or academic circulars today?",
                "hi": "नमस्ते! मैं आपका कैंपस सहायक हूँ। परीक्षा कार्यक्रम, शुल्क सूचना, छात्रवृत्ति या कॉलेज नोटिस के बारे में आप क्या जानना चाहते हैं?",
                "hinglish": "Hello! Main aapka Campus Assistant hoon. Exam dates, fee submission, scholarships ya kisi bhi circular se related aap kya poochna chahte hain?"
            }
            return {
                "draft_response": greetings_map.get(detected_lang, greetings_map["en"]),
                "final_response": greetings_map.get(detected_lang, greetings_map["en"])
            }

        # Handle out of scope queries
        if intent == "out_of_scope":
            out_of_scope_map = {
                "en": "I am specifically designed to assist with university notices, schedules, fees, exams, and academic guidelines. Please ask a campus-related question.",
                "hi": "मैं केवल विश्वविद्यालय के नोटिस, परीक्षा समय-सारणी, शुल्क और शैक्षणिक दिशा-निर्देशों में सहायता के लिए अधिकृत हूँ।",
                "hinglish": "Main sirf college notices, exams, fees aur academic queries mein help kar sakta hoon. Kripya campus se related sawaal poochein."
            }
            return {
                "draft_response": out_of_scope_map.get(detected_lang, out_of_scope_map["en"]),
                "final_response": out_of_scope_map.get(detected_lang, out_of_scope_map["en"])
            }

        # Fallback if no relevant documents passed the similarity threshold
        if not context_found or not retrieved_docs:
            fallback = FALLBACK_MESSAGES.get(detected_lang, FALLBACK_MESSAGES["en"])
            return {
                "draft_response": fallback,
                "final_response": fallback
            }

        # Format context excerpts
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_blocks.append(
                f"[Source {i}: {doc['document_title']} (Page {doc.get('page_number', 1)})]\n{doc['chunk_text']}"
            )
        formatted_context = "\n\n".join(context_blocks)

        # Format chat history (last 4 turns)
        raw_history = state.get("history", [])
        history_lines = [
            f"{turn.get('sender', 'user').capitalize()}: {turn.get('text', '')}"
            for turn in raw_history[-4:]
        ]
        formatted_history = "\n".join(history_lines) if history_lines else "No previous context."

        # Target language specification for generation
        target_lang_str = "Hindi (Devanagari)" if detected_lang == "hi" else (
            "Hinglish (Romanized conversational Hindi)" if detected_lang == "hinglish" else "English"
        )

        # Generate live IST timestamp for deadline & schedule awareness
        current_dt_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p IST")

        rag_chain = CAMPUS_RAG_PROMPT | self.llm
        try:
            result = rag_chain.invoke({
                "system_directive": SYSTEM_CAMPUS_GROUNDING_DIRECTIVE,
                "current_datetime": current_dt_ist,
                "context": formatted_context,
                "chat_history": formatted_history,
                "query": query,
                "target_language": target_lang_str
            })
            answer = getattr(result, "content", str(result)).strip()
        except Exception as e:
            print(f"[CampusChatbotWorkflow] Error during synthesis: {e}")
            answer = FALLBACK_MESSAGES.get(detected_lang, FALLBACK_MESSAGES["en"])

        return {
            "draft_response": answer,
            "final_response": answer
        }

    # -----------------------------------------------------------------------
    # Node 5: Translation & Localization
    # -----------------------------------------------------------------------
    def translate_response_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Validates output language. Translates the draft if synthesis was produced in English
        but user requires Hindi or Hinglish.
        """
        draft = state.get("draft_response", "")
        detected_lang = state.get("detected_language", "en")
        intent = state.get("intent", "")

        # Skip translation if already handled by greeting, out of scope, or fallback
        if intent in ("general_greeting", "out_of_scope") or detected_lang == "en":
            return {"final_response": draft}

        # If language is Hindi or Hinglish and response contains no target characters
        if detected_lang == "hi" and not self.lang_service.is_devanagari(draft):
            translated = self.lang_service.translate_response(draft, target_language="hi")
            return {"final_response": translated}

        return {"final_response": draft}

    # -----------------------------------------------------------------------
    # Graph Routing Logic
    # -----------------------------------------------------------------------
    def _route_after_intent(self, state: AgentState) -> str:
        """Branches directly to answer synthesis if intent doesn't need vector search."""
        intent = state.get("intent", "")
        if intent in ("general_greeting", "out_of_scope"):
            return "synthesize_answer"
        return "retrieve_context"

    def _build_graph(self):
        """Builds and compiles the StateGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("detect_language", self.detect_language_node)
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("synthesize_answer", self.synthesize_answer_node)
        workflow.add_node("translate_response", self.translate_response_node)

        # Define edge transitions
        workflow.set_entry_point("detect_language")
        workflow.add_edge("detect_language", "classify_intent")

        workflow.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "retrieve_context": "retrieve_context",
                "synthesize_answer": "synthesize_answer"
            }
        )

        workflow.add_edge("retrieve_context", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "translate_response")
        workflow.add_edge("translate_response", END)

        return workflow.compile()

    def run(self, query: str, conversation_id: str, preferred_language: str = "auto", history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Executes the compiled state graph for a student query turn.
        """
        initial_state: AgentState = {
            "query": query,
            "conversation_id": conversation_id,
            "preferred_language": preferred_language,
            "detected_language": "en",
            "is_multilingual": False,
            "translated_query_en": query,
            "intent": "campus_academics",
            "retrieved_documents": [],
            "context_found": False,
            "draft_response": "",
            "final_response": "",
            "sources": [],
            "history": history or []
        }

        return self.graph.invoke(initial_state)


# Global singleton instance
_workflow_instance: CampusChatbotWorkflow = None


def get_chatbot_workflow() -> CampusChatbotWorkflow:
    """Returns singleton instance of CampusChatbotWorkflow."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = CampusChatbotWorkflow()
    return _workflow_instance
