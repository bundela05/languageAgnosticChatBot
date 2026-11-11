import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from pypdf import PdfReader
import pdfplumber

from app.models.schemas import DocumentMetadata

DEFAULT_NOTICES_DIR = os.getenv("UPLOAD_DIR", "./documents/official_notices")


class CampusDocumentLoader:
    """
    Robust loader for college circulars, notices, and handbooks.
    Supports hybrid extraction: plain text + structured table layouts.
    """

    def __init__(self, notices_dir: str = DEFAULT_NOTICES_DIR):
        self.notices_dir = Path(notices_dir)
        self.notices_dir.mkdir(parents=True, exist_ok=True)

    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Extracts textual content and tabular data page-by-page from a single PDF.
        """
        documents: List[Document] = []
        filename = file_path.name
        category = self._infer_category(filename)

        try:
            # First pass: use pdfplumber for table preservation and layout extraction
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    extracted_text = ""
                    
                    # 1. Extract layout-aware plain text
                    page_text = page.extract_text(layout=False) or ""
                    
                    # 2. Extract and format structured tables (timetables, fee slabs, dates)
                    tables = page.extract_tables()
                    table_str = ""
                    if tables:
                        for table in tables:
                            # Filter empty rows/cells and format into structured markdown rows
                            cleaned_rows = [
                                " | ".join([str(cell).strip() if cell is not None else "" for cell in row])
                                for row in table if any(row)
                            ]
                            if cleaned_rows:
                                table_str += "\n\n[TABLE DATA]\n" + "\n".join(cleaned_rows) + "\n[/TABLE DATA]\n"

                    combined_text = (page_text + "\n" + table_str).strip()

                    # Fallback to pypdf if pdfplumber yields empty output for this page
                    if not combined_text:
                        combined_text = self._fallback_pypdf_page(file_path, page_idx - 1)

                    if combined_text:
                        metadata = DocumentMetadata(
                            source=filename,
                            page=page_idx,
                            category=category
                        ).model_dump()
                        
                        documents.append(
                            Document(
                                page_content=combined_text,
                                metadata=metadata
                            )
                        )

        except Exception as e:
            # Global fallback to standard pypdf extraction if pdfplumber fails entirely
            print(f"[CampusDocumentLoader] pdfplumber error on {filename}: {e}. Retrying with pypdf...")
            documents = self._fallback_pypdf_all(file_path, category)

        return documents

    def load_all_notices(self) -> List[Document]:
        """
        Recursively scans documents/official_notices/ for PDF and text files.
        """
        all_documents: List[Document] = []
        valid_extensions = {".pdf", ".txt"}

        for file_path in self.notices_dir.rglob("*"):
            if file_path.suffix.lower() in valid_extensions and file_path.is_file():
                if file_path.suffix.lower() == ".pdf":
                    docs = self.load_pdf(file_path)
                else:
                    docs = self._load_txt(file_path)
                all_documents.extend(docs)

        return all_documents

    def _load_txt(self, file_path: Path) -> List[Document]:
        """Loads simple text circulars."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    metadata = DocumentMetadata(
                        source=file_path.name,
                        page=1,
                        category=self._infer_category(file_path.name)
                    ).model_dump()
                    return [Document(page_content=content, metadata=metadata)]
        except Exception as e:
            print(f"[CampusDocumentLoader] Error reading text file {file_path.name}: {e}")
        return []

    def _fallback_pypdf_page(self, file_path: Path, page_index: int) -> str:
        """Extracts text for an individual page using pypdf."""
        try:
            reader = PdfReader(str(file_path))
            if 0 <= page_index < len(reader.pages):
                return reader.pages[page_index].extract_text() or ""
        except Exception:
            pass
        return ""

    def _fallback_pypdf_all(self, file_path: Path, category: str) -> List[Document]:
        """Full fallback extracting all pages via pypdf."""
        docs: List[Document] = []
        try:
            reader = PdfReader(str(file_path))
            for page_idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    metadata = DocumentMetadata(
                        source=file_path.name,
                        page=page_idx,
                        category=category
                    ).model_dump()
                    docs.append(Document(page_content=text.strip(), metadata=metadata))
        except Exception as e:
            print(f"[CampusDocumentLoader] pypdf failed completely on {file_path.name}: {e}")
        return docs

    @staticmethod
    def _infer_category(filename: str) -> str:
        """
        Infers departmental classification based on filename keywords.
        """
        name_lower = filename.lower()
        if any(k in name_lower for k in ["fee", "dues", "tuition", "fine"]):
            return "fees"
        if any(k in name_lower for k in ["exam", "time_table", "schedule", "revaluation", "result"]):
            return "exams"
        if any(k in name_lower for k in ["scholarship", "medhavi", "post_matric"]):
            return "scholarships"
        if any(k in name_lower for k in ["admission", "counseling", "seat", "branch"]):
            return "admissions"
        if any(k in name_lower for k in ["holiday", "calendar", "event", "hostel", "bus"]):
            return "campus_life"
        return "general"
