import json
import os
import re
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from app.chatbot.prompts import INTENT_CLASSIFICATION_PROMPT

load_dotenv()


class CampusIntentClassifier:
    """
    Classifies student queries into actionable campus administrative domains,
    detects non-campus queries, and generates category metadata filters for vector search.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        api_key = os.getenv("GROQ_API_KEY", "")
        model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        self.llm = llm or ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0
        )

    @staticmethod
    def rule_based_check(query: str) -> Optional[Dict[str, Any]]:
        """
        Fast-path heuristic evaluation for common short greetings and standard keywords
        to bypass LLM inference latency.
        """
        cleaned = query.strip().lower()

        # 1. Quick conversational greetings & exits
        greetings = {"hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening"}
        if cleaned in greetings:
            return {
                "intent": "general_greeting",
                "category_filter": "general",
                "confidence": 1.0,
                "is_greeting": True
            }

        closings = {"thank you", "thanks", "dhanyawad", "bye", "goodbye", "ok thanks"}
        if cleaned in closings:
            return {
                "intent": "general_closing",
                "category_filter": "general",
                "confidence": 1.0,
                "is_greeting": True
            }

        # 2. Direct keyword mapping for high-frequency terms
        if any(w in cleaned for w in ["fee", "fees", "fine", "dues", "installment", "challan"]):
            return {"intent": "campus_fees", "category_filter": "fees", "confidence": 0.95, "is_greeting": False}

        if any(w in cleaned for w in ["exam", "time table", "timetable", "admit card", "hall ticket", "reval", "revaluation"]):
            return {"intent": "campus_exams", "category_filter": "exams", "confidence": 0.95, "is_greeting": False}

        if any(w in cleaned for w in ["scholarship", "medhavi", "post matric", "nsp"]):
            return {"intent": "campus_scholarship", "category_filter": "scholarships", "confidence": 0.95, "is_greeting": False}

        return None

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Evaluates query intent using rule heuristics first, falling back to LLM JSON parsing.
        Returns a dict containing:
          - intent: str
          - category_filter: str ('fees', 'exams', 'scholarships', 'admissions', 'general')
          - confidence: float
          - is_greeting: bool
        """
        # Fast path check
        fast_result = self.rule_based_check(query)
        if fast_result:
            return fast_result

        try:
            chain = INTENT_CLASSIFICATION_PROMPT | self.llm
            response = chain.invoke({"query": query})
            content = getattr(response, "content", str(response)).strip()

            # Extract JSON block using regex if wrapped in markdown formatting
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                intent = parsed.get("intent", "campus_academics")
                category_filter = parsed.get("category_filter", "general")
                confidence = float(parsed.get("confidence", 0.85))
                return {
                    "intent": intent,
                    "category_filter": category_filter,
                    "confidence": confidence,
                    "is_greeting": intent in ("general_greeting", "general_closing")
                }
        except Exception as e:
            print(f"[CampusIntentClassifier] Error in classification: {e}. Defaulting to general campus query.")

        # Default fallback
        return {
            "intent": "campus_academics",
            "category_filter": "general",
            "confidence": 0.50,
            "is_greeting": False
        }


# Singleton accessor
_classifier_instance: Optional[CampusIntentClassifier] = None


def get_intent_classifier() -> CampusIntentClassifier:
    """Returns singleton instance of CampusIntentClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = CampusIntentClassifier()
    return _classifier_instance
