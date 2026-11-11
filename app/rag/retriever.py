import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.database.database import VectorStoreManager
from app.models.schemas import SourceCitation
from app.rag.embeddings import MultilingualEmbeddingService
from app.rag.loader import CampusDocumentLoader
from app.rag.splitter import CampusDocumentSplitter


class CampusRetriever:
    """
    Manages indexing and similarity search over campus circulars and notices.
    Coordinates between ChromaDB, multilingual embeddings, and document processing.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        top_k: int = 4,
        similarity_threshold: float = 0.35
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        target_collection = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "campus_documents")
        
        self.collection = VectorStoreManager.get_collection(target_collection)
        self.embedding_service = MultilingualEmbeddingService.get_instance()
        self.loader = CampusDocumentLoader()
        self.splitter = CampusDocumentSplitter()

    def index_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Parses, splits, embeds, and indexes a single PDF/text document into ChromaDB.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Remove existing records for this file to prevent duplicate chunks on re-upload
        self.delete_document_records(file_path.name)

        # 1. Parse document pages & tables
        raw_documents = self.loader.load_pdf(file_path) if file_path.suffix.lower() == ".pdf" else self.loader._load_txt(file_path)
        if not raw_documents:
            return {"filename": file_path.name, "pages": 0, "chunks": 0, "status": "empty"}

        # 2. Chunk text while preserving tabular structures
        chunks = self.splitter.split_documents(raw_documents)
        if not chunks:
            return {"filename": file_path.name, "pages": len(raw_documents), "chunks": 0, "status": "no_chunks"}

        # 3. Vectorize chunks using multilingual embeddings
        contents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(contents)
        chunk_ids = [f"{file_path.name}_{chunk.metadata.get('page', 1)}_{i}_{uuid.uuid4().hex[:6]}" for i, chunk in enumerate(chunks)]

        # 4. Insert into persistent ChromaDB collection
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )

        return {
            "filename": file_path.name,
            "pages": len(raw_documents),
            "chunks": len(chunks),
            "status": "indexed"
        }

    def index_all_documents(self) -> Dict[str, Any]:
        """
        Scans documents/official_notices/ and indexes all found documents.
        """
        all_docs = self.loader.load_all_notices()
        if not all_docs:
            return {"total_documents": 0, "total_chunks": 0, "status": "empty_directory"}

        chunks = self.splitter.split_documents(all_docs)
        if not chunks:
            return {"total_documents": len(all_docs), "total_chunks": 0, "status": "no_chunks"}

        contents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(contents)
        chunk_ids = [f"doc_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )

        return {
            "total_documents": len(all_docs),
            "total_chunks": len(chunks),
            "status": "complete"
        }

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SourceCitation]:
        """
        Performs vector similarity search for a query and filters results by similarity threshold.
        """
        k = top_k or self.top_k
        query_embedding = self.embedding_service.embed_query(query)

        where_filter = {"category": category} if category and category != "general" else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        citations: List[SourceCitation] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return citations

        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        distances = results["distances"][0] if results["distances"] else [1.0] * len(documents)

        for doc_text, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance range: [0, 2], where 0 is identical and 1 is orthogonal.
            # Convert cosine distance to cosine similarity: sim = 1 - distance
            similarity = 1.0 - dist
            if similarity >= self.similarity_threshold:
                citations.append(
                    SourceCitation(
                        document_title=meta.get("source", "Campus Notice"),
                        page_number=meta.get("page", 1),
                        chunk_text=doc_text,
                        relevance_score=round(similarity, 4)
                    )
                )

        return citations

    def delete_document_records(self, filename: str) -> None:
        """
        Removes all indexed chunks associated with a specific file from ChromaDB.
        """
        try:
            self.collection.delete(where={"source": filename})
        except Exception:
            pass


# Global accessor instance
_retriever_instance: Optional[CampusRetriever] = None

def get_campus_retriever() -> CampusRetriever:
    """Returns a singleton instance of CampusRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = CampusRetriever()
    return _retriever_instance
