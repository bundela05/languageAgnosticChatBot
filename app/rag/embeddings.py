import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

# Multilingual embedding model identifier
# 'paraphrase-multilingual-MiniLM-L12-v2' maps 50+ languages (including English and Hindi)
# into a unified 384-dimensional vector space.
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class MultilingualEmbeddingService(Embeddings):
    """
    Embedding service implementing LangChain's Embeddings interface.
    Provides cross-lingual representations for document chunks and user queries.
    """
    _instance: Optional["MultilingualEmbeddingService"] = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        # Loads on CPU or Apple Silicon / CUDA automatically
        self.client = SentenceTransformer(model_name)

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> "MultilingualEmbeddingService":
        """Singleton accessor to prevent reloading weights into memory repeatedly."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings for ingested text chunks.
        Normalizes vectors for accurate cosine similarity calculation.
        """
        if not texts:
            return []
        embeddings = self.client.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single student search query.
        """
        if not text.strip():
            return [0.0] * self.client.get_sentence_embedding_dimension()
        embedding = self.client.encode(
            text, 
            normalize_embeddings=True
        )
        return embedding.tolist()


def get_embedding_function():
    """
    Returns a callable embedding function compatible with ChromaDB's native collection API.
    """
    service = MultilingualEmbeddingService.get_instance()

    class ChromaEmbeddingAdapter:
        def __call__(self, input: List[str]) -> List[List[float]]:
            return service.embed_documents(input)

    return ChromaEmbeddingAdapter()
