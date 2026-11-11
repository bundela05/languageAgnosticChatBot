from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CampusDocumentSplitter:
    """
    Splits loaded documents and notices into semantic, overlapping chunks.
    Preserves table data blocks and structural headers for accurate retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Priority order: Section boundaries, table block delimiters, 
        # paragraph breaks, sentence stops, clause breaks, and spaces.
        self.separators = separators or [
            "\n[/TABLE DATA]\n",
            "\n\n",
            "\n",
            "। ",  # Hindi Purna Viram sentence boundary
            ". ",
            "? ",
            "! ",
            "; ",
            " ",
            ""
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of parsed documents into smaller chunks and enriches
        each chunk's metadata with chunk index and character length.
        """
        if not documents:
            return []

        raw_chunks = self.text_splitter.split_documents(documents)
        enriched_chunks: List[Document] = []

        chunk_counter = 0
        for chunk in raw_chunks:
            cleaned_content = chunk.page_content.strip()
            # Ignore ultra-short fragments or residual empty lines
            if len(cleaned_content) < 20:
                continue

            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = chunk_counter
            metadata["char_length"] = len(cleaned_content)
            
            enriched_chunks.append(
                Document(
                    page_content=cleaned_content,
                    metadata=metadata
                )
            )
            chunk_counter += 1

        return enriched_chunks

    def split_text(self, text: str, base_metadata: Optional[dict] = None) -> List[Document]:
        """
        Helper method to split raw text directly into Document chunks.
        """
        if not text.strip():
            return []
            
        base_meta = base_metadata or {}
        doc = Document(page_content=text, metadata=base_meta)
        return self.split_documents([doc])
