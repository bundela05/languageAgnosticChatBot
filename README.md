# languageAgnosticChatBot

An AI-driven, multilingual, document-grounded conversational assistant built for campus queries and institutional circulars. Powered by **FastAPI**, **LangChain**, **LangGraph**, **ChromaDB**, and **SQLite**, the system enables students and faculty to query official PDF notices, exam schedules, fee deadlines, and academic guidelines in **English**, **Hindi (Devanagari)**, or **Hinglish** with zero-hallucination citation grounding.

---

## ✨ Features

- **Multilingual & Hinglish Support:** Automatically detects incoming English, Hindi (Devanagari), and colloquial Hinglish queries, normalizes them into canonical English for cross-lingual retrieval, and delivers localized responses.
- **LangGraph State Machine Workflow:** Orchestrates query processing through a structured pipeline (`Language Detection` -> `Intent Classification` -> `Cross-Lingual RAG Retrieval` -> `Grounded Synthesis` -> `Localization`).
- **Hybrid PDF & Table Parsing:** Uses `pdfplumber` and `pypdf` to extract both layout-aware text and structured tables (such as fee slabs and academic timelines) from official campus circulars.
- **Dense Multilingual Vector Search:** Indexes document chunks into **ChromaDB** using `paraphrase-multilingual-MiniLM-L12-v2` embeddings with cosine similarity thresholding to prevent hallucination.
- **Persistent Session & Audit Memory:** Stores multi-turn conversation history, indexed document metadata, and system audit logs in **SQLite**.
- **Glassmorphic Web UI & Admin Drawer:** Includes a responsive HTML5/CSS3/JS frontend featuring Markdown table rendering, expandable source citations, language overrides, and a drag-and-drop notice upload panel.

---

## 🛠️ Tech Stack

- **Backend Framework:** Python 3.9+, FastAPI, Uvicorn, Pydantic v2
- **AI & Agent Orchestration:** LangChain, LangGraph, Groq (`llama-3.3-70b-versatile`)
- **Vector Store & Embeddings:** ChromaDB, Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Relational Storage:** SQLite3
- **Document Parsing & NLP:** `pdfplumber`, `pypdf`, `langdetect`
- **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism), JavaScript, Marked.js, Lucide Icons
- **Testing:** Pytest, FastAPI `TestClient`, HTTPX

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/bundela05/languageAgnosticChatBot.git
cd languageAgnosticChatBot
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate on macOS / Linux:
source venv/bin/activate

# Activate on Windows:
# venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root directory:
```env
HOST=0.0.0.0
PORT=8000
DEBUG=True

# LLM Configuration
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile

# Storage & Paths
SQLITE_DB_PATH=./campus_chatbot.db
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=campus_documents
UPLOAD_DIR=./documents/official_notices
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 4. Add Official Campus Notices
Place official PDF or TXT circulars inside `documents/official_notices/`. The server automatically syncs and indexes files in this directory upon startup, or you can upload and manage documents directly from the web UI sidebar.

### 5. Launch the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to **`http://localhost:8000`** to access the chat interface and document manager.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the frontend single-page web interface. |
| `GET` | `/api/health` | Verifies server operational status and directory mounts. |
| `POST` | `/api/chat` | Processes a user query through the LangGraph pipeline and returns verified citations. |
| `POST` | `/api/chat/stream` | Streams real-time response chunks via Server-Sent Events (SSE). |
| `GET` | `/api/chat/history/{conversation_id}` | Retrieves stored multi-turn message history for a session. |
| `POST` | `/api/documents/upload` | Uploads, parses, chunks, and indexes a new PDF/TXT circular into ChromaDB. |
| `POST` | `/api/documents/reindex-all` | Re-indexes all files present in `documents/official_notices/`. |
| `GET` | `/api/documents/list` | Lists all tracked notices with page counts and chunk statistics. |
| `DELETE` | `/api/documents/{filename}` | Removes a notice from disk, SQLite, and ChromaDB vector memory. |

---

## 🧪 Running Tests

Run the unit and integration test suite covering language detection, RAG chunking/retrieval, and API endpoints:
```bash
pytest -v
```

---

## 📁 Project Structure

```text
languageAgnosticChatBot/
│
├── app/
│   ├── main.py                  # FastAPI entry point, CORS, lifespan sync, static UI mount
│   │
│   ├── api/
│   │   ├── chat.py              # Synchronous & streaming chat endpoints + history
│   │   └── documents.py         # Circular upload, re-indexing, listing & deletion APIs
│   │
│   ├── chatbot/
│   │   ├── graph.py             # Compiled LangGraph StateGraph orchestration
│   │   ├── prompts.py           # Grounding directives, normalization & translation prompts
│   │   ├── language.py          # English/Hindi/Hinglish detection & translation helpers
│   │   └── intent.py            # Rule + LLM intent classifier & department router
│   │
│   ├── rag/
│   │   ├── loader.py            # Hybrid PDF (pdfplumber + pypdf) & TXT document loader
│   │   ├── splitter.py          # Table-preserving recursive character splitter
│   │   ├── embeddings.py        # Multilingual SentenceTransformer embedding wrapper
│   │   └── retriever.py         # ChromaDB indexing & cosine similarity search interface
│   │
│   ├── database/
│   │   └── database.py          # SQLite schema/queries & persistent ChromaDB client
│   │
│   └── models/
│       └── schemas.py           # Pydantic request/response models & LangGraph AgentState
│
├── documents/
│   └── official_notices/        # Repository for official campus PDF/TXT circulars
│
├── frontend/
│   ├── index.html               # Single-page chat UI & notice management drawer
│   ├── style.css                # Glassmorphic dark theme & responsive layout
│   └── script.js                # Session state, fetch/SSE calls, Markdown & citation rendering
│
├── tests/
│   ├── test_chat.py             # FastAPI TestClient integration tests
│   ├── test_rag.py              # Loader, splitter & ChromaDB retrieval unit tests
│   └── test_language.py         # Language detection, Hinglish & translation unit tests
│
├── .env                         # Environment variables & API keys (git-ignored)
├── .gitignore
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss the changes you would like to make, or submit a pull request directly.