import os
import re
from typing import Dict, Tuple
from dotenv import load_dotenv
from langdetect import DetectorFactory, detect
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from app.chatbot.prompts import QUERY_NORMALIZATION_PROMPT, TRANSLATION_PROMPT

load_dotenv()

# Enforce deterministic results for langdetect
DetectorFactory.seed = 0

# Common Hinglish marker words frequently used by students in campus contexts
HINGLISH_KEYWORDS = {
    "kya", "kab", "kaise", "kahan", "kitna", "kitni", "kyu", "kyun", "kisko",
    "batao", "bataiye", "dekhna", "karein", "karna", "hoga", "hogi", "hote",
    "hai", "hain", "tha", "thi", "the", "mera", "meri", "mere", "apna", "apni",
    "fees", "form", "date", "last", "submit", "mil", "sakta", "sakti", "chahiye",
    "admission", "exam", "result", "timetable", "holiday", "chhutti", "sir", "mam"
}


class LanguageService:
    """
    Handles language detection, script classification, and cross-lingual translation.
    Tailored for campus queries submitted in English, Devanagari Hindi, or Hinglish.
    """

    def __init__(self, llm: BaseChatModel = None):
        api_key = os.getenv("GROQ_API_KEY", "")
        model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        self.llm = llm or ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0
        )

    @staticmethod
    def is_devanagari(text: str) -> bool:
        """Checks if text contains Devanagari Unicode script characters."""
        return bool(re.search(r"[\u0900-\u097F]", text))

    @classmethod
    def is_hinglish(cls, text: str) -> bool:
        """
        Determines whether Roman-script text represents colloquial Hinglish
        based on token frequency against common Hindi stop/query markers.
        """
        # If written in Devanagari, it's native Hindi, not Hinglish
        if cls.is_devanagari(text):
            return False

        # Tokenize Roman words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return False

        hinglish_match_count = sum(1 for w in words if w in HINGLISH_KEYWORDS)
        match_ratio = hinglish_match_count / len(words)

        # Flag as Hinglish if at least 2 key markers or > 20% match ratio
        return hinglish_match_count >= 2 or (len(words) >= 3 and match_ratio >= 0.20)

    def detect_language(self, text: str, preferred_language: str = "auto") -> Tuple[str, str]:
        """
        Detects query language. Returns a tuple: (language_code, language_label).
        Examples:
          - ("en", "English")
          - ("hi", "Hindi")
          - ("hinglish", "Hinglish")
        """
        if preferred_language and preferred_language != "auto":
            labels = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
            return preferred_language, labels.get(preferred_language, "English")

        clean_text = text.strip()
        if not clean_text:
            return "en", "English"

        # 1. Check Devanagari Unicode block
        if self.is_devanagari(clean_text):
            return "hi", "Hindi"

        # 2. Check Romanized Hindi (Hinglish)
        if self.is_hinglish(clean_text):
            return "hinglish", "Hinglish"

        # 3. Fallback to statistical langdetect
        try:
            detected = detect(clean_text)
            if detected == "hi":
                return "hi", "Hindi"
            return "en", "English"
        except Exception:
            return "en", "English"

    def normalize_query_for_retrieval(self, query: str, detected_language: str) -> str:
        """
        Normalizes Hindi or Hinglish queries into canonical English to maximize
        vector retrieval accuracy against formal English circulars and notices.
        """
        # If query is purely English, no translation/normalization step needed
        if detected_language == "en":
            return query.strip()

        try:
            chain = QUERY_NORMALIZATION_PROMPT | self.llm
            result = chain.invoke({
                "query": query,
                "detected_language": detected_language
            })
            normalized = getattr(result, "content", str(result)).strip()
            return normalized if normalized else query
        except Exception as e:
            print(f"[LanguageService] Query normalization fallback due to error: {e}")
            return query

    def translate_response(self, text: str, target_language: str) -> str:
        """
        Translates a synthesized response to the target language (Hindi or Hinglish),
        preserving dates, tables, circular numbers, and fee amounts.
        """
        if target_language in ("en", "English"):
            return text

        try:
            chain = TRANSLATION_PROMPT | self.llm
            result = chain.invoke({
                "draft_response": text,
                "target_language": target_language
            })
            translated = getattr(result, "content", str(result)).strip()
            return translated if translated else text
        except Exception as e:
            print(f"[LanguageService] Translation error: {e}. Returning original.")
            return text


# Global singleton instance
_lang_service: LanguageService = None


def get_language_service() -> LanguageService:
    """Returns singleton instance of LanguageService."""
    global _lang_service
    if _lang_service is None:
        _lang_service = LanguageService()
    return _lang_service
