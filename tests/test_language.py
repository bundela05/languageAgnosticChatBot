import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.chatbot.language import LanguageService


@pytest.fixture
def mock_llm():
    """Creates a mocked BaseChatModel to avoid external API calls during unit tests."""
    llm = MagicMock()
    return llm


@pytest.fixture
def language_service(mock_llm):
    """Initializes LanguageService with a mocked LLM backend."""
    return LanguageService(llm=mock_llm)


# ---------------------------------------------------------------------------
# 1. Script & Heuristic Detection Tests (SRS T-01 & T-02)
# ---------------------------------------------------------------------------

class TestLanguageDetection:

    def test_detect_english_query(self, language_service):
        query = "When is the last date for B.Tech exam form submission?"
        code, label = language_service.detect_language(query)
        assert code == "en"
        assert label == "English"

    def test_detect_hindi_devanagari_query(self, language_service):
        query = "परीक्षा शुल्क जमा करने की अंतिम तिथि क्या है?"
        code, label = language_service.detect_language(query)
        assert code == "hi"
        assert label == "Hindi"

    def test_detect_hinglish_mixed_query(self, language_service):
        query = "Semester exam fees submit karne ki last date kya hai?"
        code, label = language_service.detect_language(query)
        assert code == "hinglish"
        assert label == "Hinglish"

    def test_detect_hinglish_short_interrogative(self, language_service):
        query = "Scholarship form kab aayega sir?"
        code, label = language_service.detect_language(query)
        assert code == "hinglish"
        assert label == "Hinglish"

    def test_explicit_preferred_language_override(self, language_service):
        # User types in English, but forces Hindi output preference
        query = "Show me the holiday list."
        code, label = language_service.detect_language(query, preferred_language="hi")
        assert code == "hi"
        assert label == "Hindi"

    def test_empty_or_whitespace_defaults_to_english(self, language_service):
        code, label = language_service.detect_language("   ")
        assert code == "en"
        assert label == "English"


# ---------------------------------------------------------------------------
# 2. Script Helper Unit Tests
# ---------------------------------------------------------------------------

class TestScriptHelpers:

    def test_is_devanagari_true(self):
        assert LanguageService.is_devanagari("छात्रवृत्ति सूचना 2026") is True

    def test_is_devanagari_false_for_roman(self):
        assert LanguageService.is_devanagari("Scholarship notice 2026") is False

    def test_is_hinglish_false_when_devanagari_present(self):
        # Even if mixed with English words, Devanagari script should classify as Hindi, not Hinglish
        assert LanguageService.is_hinglish("मेरा exam कब है?") is False


# ---------------------------------------------------------------------------
# 3. Query Normalization & Translation Pipeline Tests
# ---------------------------------------------------------------------------

class TestNormalizationAndTranslation:

    def test_normalize_english_skips_llm(self, language_service, mock_llm):
        query = "What are the revaluation guidelines?"
        normalized = language_service.normalize_query_for_retrieval(query, detected_language="en")
        assert normalized == query
        mock_llm.invoke.assert_not_called()

    def test_normalize_hinglish_invokes_llm(self, language_service):
        with patch("app.chatbot.language.QUERY_NORMALIZATION_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(content="What is the deadline for submitting semester examination fees?")
            mock_prompt.__or__.return_value = mock_chain

            result = language_service.normalize_query_for_retrieval(
                "Semester exam fees bharne ki last date kya hai?",
                detected_language="hinglish"
            )
            assert "examination fees" in result.lower()

    def test_translate_response_skips_if_target_is_english(self, language_service, mock_llm):
        draft = "The last date for synopsis submission is 5-Sep-2026."
        output = language_service.translate_response(draft, target_language="en")
        assert output == draft
        mock_llm.invoke.assert_not_called()

    def test_translate_response_to_hindi(self, language_service):
        with patch("app.chatbot.language.TRANSLATION_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(content="सिनॉप्सिस जमा करने की अंतिम तिथि 5-Sep-2026 है।")
            mock_prompt.__or__.return_value = mock_chain

            result = language_service.translate_response(
                "The last date for synopsis submission is 5-Sep-2026.",
                target_language="hi"
            )
            assert "5-Sep-2026" in result
            assert LanguageService.is_devanagari(result) is True
