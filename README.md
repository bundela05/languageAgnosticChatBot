# languageAgnosticChatBot
A lightweight, language-agnostic chatbot engine supporting multilingual conversational AI across diverse frameworks, backends, and interfaces.

---

## ✨ Features

- **Language-Agnostic Design:** Processes input dynamically using multilingual tokenizers and embeddings.
- **Pluggable Architecture:** Easy integration with multiple LLM backends (OpenAI, Hugging Face, Ollama, local models).
- **Fast & Scalable:** Modular API layer built for rapid deployment and minimal inference latency.
- **Context-Aware:** Efficient conversation memory retention across session turns.

---

## 🚀 Quick Start

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/bundela05/polyglot-chatbot.git
cd languageAgnosticChatBot-chatbot
\`\`\`

### 2. Set Up Environment
\`\`\`bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 3. Run the Chatbot
\`\`\`bash
python main.py
\`\`\`

## 📁 Project Structure

```text
languageAgnosticChatBot/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── chat.py
│   │   └── documents.py
│   │
│   ├── chatbot/
│   │   ├── graph.py
│   │   ├── prompts.py
│   │   ├── language.py
│   │   └── intent.py
│   │
│   ├── rag/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   └── retriever.py
│   │
│   ├── database/
│   │   └── database.py
│   │
│   └── models/
│       └── schemas.py
│
├── documents/
│   └── official_notices/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── tests/
│   ├── test_chat.py
│   ├── test_rag.py
│   └── test_language.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss the changes you would like to make, or submit a pull request directly.
