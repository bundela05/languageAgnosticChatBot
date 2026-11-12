/**
 * Language-Agnostic Campus Query Assistant
 * Frontend Controller: Handles session persistence, chat streaming,
 * document ingestion, and dynamic Markdown/Citation rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------------------------------------------------
    // 1. DOM Elements & State
    // -----------------------------------------------------------------------
    const chatFlow = document.getElementById("chatFlow");
    const messagesContainer = document.getElementById("messagesContainer");
    const welcomeScreen = document.getElementById("welcomeScreen");
    const chatForm = document.getElementById("chatForm");
    const queryInput = document.getElementById("queryInput");
    const sendBtn = document.getElementById("sendBtn");
    const languageSelect = document.getElementById("languageSelect");
    const newChatBtn = document.getElementById("newChatBtn");

    // Sidebar & Upload Elements
    const sidebar = document.getElementById("sidebar");
    const menuToggleBtn = document.getElementById("menuToggleBtn");
    const closeSidebarBtn = document.getElementById("closeSidebarBtn");
    const uploadForm = document.getElementById("uploadForm");
    const noticeFileInput = document.getElementById("noticeFileInput");
    const dropzoneLabel = document.getElementById("dropzoneLabel");
    const dropzone = document.getElementById("dropzone");
    const uploadFeedback = document.getElementById("uploadFeedback");
    const indexedDocList = document.getElementById("indexedDocList");
    const refreshDocsBtn = document.getElementById("refreshDocsBtn");

    // Configure marked.js options
    if (window.marked) {
        marked.setOptions({
            breaks: true,
            gfm: true
        });
    }

    // Initialize or restore conversation UUID
    let conversationId = localStorage.getItem("campus_chat_convo_id");
    if (!conversationId) {
        conversationId = generateUUID();
        localStorage.setItem("campus_chat_convo_id", conversationId);
    }

    let isProcessing = false;

    // Refresh icons on load
    refreshLucideIcons();

    // -----------------------------------------------------------------------
    // 2. Chat Execution & Rendering Logic
    // -----------------------------------------------------------------------

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query || isProcessing) return;

        // Hide empty welcome screen on first interaction
        if (welcomeScreen) {
            welcomeScreen.style.display = "none";
        }

        // Render user message bubble
        appendMessage({
            sender: "user",
            text: query,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        });

        queryInput.value = "";
        autoResizeTextarea();
        isProcessing = true;
        sendBtn.disabled = true;

        // Show typing indicator
        const typingIndicator = createTypingIndicator();
        chatFlow.appendChild(typingIndicator);
        scrollToBottom();

        const preferredLang = languageSelect.value;

        try {
            // Initiate standard chat invocation
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: query,
                    conversation_id: conversationId,
                    preferred_language: preferredLang
                })
            });

            typingIndicator.remove();

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned error ${response.status}`);
            }

            const data = await response.json();

            // Append assistant response
            appendMessage({
                sender: "assistant",
                text: data.response,
                detectedLanguage: data.detected_language,
                intent: data.intent,
                sources: data.sources || [],
                timestamp: new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            });

        } catch (error) {
            if (typingIndicator) typingIndicator.remove();
            appendMessage({
                sender: "assistant",
                text: `⚠️ **Error:** Unable to complete your request. ${error.message}`,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            });
        } finally {
            isProcessing = false;
            sendBtn.disabled = false;
            queryInput.focus();
            scrollToBottom();
        }
    });

    function appendMessage({ sender, text, detectedLanguage, intent, sources = [], timestamp }) {
        const row = document.createElement("div");
        row.className = `message-row ${sender === "user" ? "user" : "bot"}`;

        if (sender === "user") {
            row.innerHTML = `
                <div class="message-body">
                    <div class="message-bubble">${escapeHTML(text)}</div>
                    <div class="bubble-meta" style="justify-content: flex-end;">
                        <span>${timestamp}</span>
                    </div>
                </div>
            `;
        } else {
            // Render markdown content
            const renderedMarkdown = window.marked ? marked.parse(text) : escapeHTML(text);

            // Build sources citations if present
            let sourcesHTML = "";
            if (sources && sources.length > 0) {
                const sourcePills = sources.map((s, idx) => {
                    const score = s.relevance_score ? `(${(s.relevance_score * 100).toFixed(0)}% match)` : "";
                    return `
                        <div class="source-badge" title="${escapeHTML(s.chunk_text.slice(0, 150))}...">
                            <i data-lucide="bookmark"></i>
                            <span>${escapeHTML(s.document_title)} (Pg. ${s.page_number || 1}) ${score}</span>
                        </div>
                    `;
                }).join("");

                sourcesHTML = `
                    <div class="sources-container">
                        <span style="font-size: 0.72rem; color: var(--text-muted); font-weight: 600;">VERIFIED CITATIONS:</span>
                        ${sourcePills}
                    </div>
                `;
            }

            // Build metadata badges
            let metaBadges = "";
            if (detectedLanguage) {
                metaBadges += `<span class="lang-pill">${detectedLanguage.toUpperCase()}</span>`;
            }
            if (intent && intent !== "campus_academics") {
                metaBadges += `<span class="intent-pill">${intent.replace("campus_", "")}</span>`;
            }

            row.innerHTML = `
                <div class="avatar bot-avatar">
                    <i data-lucide="bot"></i>
                </div>
                <div class="message-body">
                    <div class="message-bubble markdown-body">
                        ${renderedMarkdown}
                        ${sourcesHTML}
                    </div>
                    <div class="bubble-meta">
                        ${metaBadges}
                        <span>${timestamp}</span>
                    </div>
                </div>
            `;
        }

        chatFlow.appendChild(row);
        refreshLucideIcons();
        scrollToBottom();
    }

    function createTypingIndicator() {
        const row = document.createElement("div");
        row.className = "message-row bot typing-row";
        row.innerHTML = `
            <div class="avatar bot-avatar">
                <i data-lucide="bot"></i>
            </div>
            <div class="message-body">
                <div class="message-bubble typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        refreshLucideIcons();
        return row;
    }

    // -----------------------------------------------------------------------
    // 3. Document Ingestion & Repository Sync
    // -----------------------------------------------------------------------

    async function loadIndexedDocuments() {
        try {
            const res = await fetch("/api/documents/list");
            if (!res.ok) throw new Error("Failed to fetch documents");
            const docs = await res.json();

            indexedDocList.innerHTML = "";
            if (!docs || docs.length === 0) {
                indexedDocList.innerHTML = `<li class="empty-doc-item">No official notices indexed yet.</li>`;
                return;
            }

            docs.forEach((doc) => {
                const li = document.createElement("li");
                li.className = "doc-item";
                li.innerHTML = `
                    <div class="doc-info">
                        <span class="doc-title" title="${escapeHTML(doc.filename)}">${escapeHTML(doc.filename)}</span>
                        <span class="doc-meta">${doc.total_pages} page(s) • ${doc.chunks_count} chunks</span>
                    </div>
                    <button class="doc-delete-btn" data-filename="${escapeHTML(doc.filename)}" title="Delete Notice">
                        <i data-lucide="trash-2"></i>
                    </button>
                `;
                indexedDocList.appendChild(li);
            });

            // Attach delete listeners
            document.querySelectorAll(".doc-delete-btn").forEach((btn) => {
                btn.addEventListener("click", () => handleDeleteDocument(btn.dataset.filename));
            });

            refreshLucideIcons();
        } catch (err) {
            indexedDocList.innerHTML = `<li class="empty-doc-item" style="color: var(--accent-rose);">Error loading documents</li>`;
        }
    }

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = noticeFileInput.files[0];
        if (!file) {
            showFeedback("Please select a PDF or TXT file to upload.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        const submitBtn = document.getElementById("uploadSubmitBtn");
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span>Indexing...</span>`;

        try {
            const res = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Upload failed");

            showFeedback(`✓ Indexed '${data.filename}' (${data.chunks_created} chunks)`, "success");
            uploadForm.reset();
            dropzoneLabel.textContent = "Click or drag PDF here";
            loadIndexedDocuments();
        } catch (err) {
            showFeedback(`Failed: ${err.message}`, "error");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<span>Index Document</span>`;
        }
    });

    async function handleDeleteDocument(filename) {
        if (!confirm(`Are you sure you want to delete and purge '${filename}' from vector memory?`)) return;

        try {
            const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
                method: "DELETE"
            });
            if (!res.ok) throw new Error("Delete failed");
            loadIndexedDocuments();
        } catch (err) {
            alert(`Error deleting notice: ${err.message}`);
        }
    }

    function showFeedback(msg, type) {
        uploadFeedback.className = `feedback-msg ${type}`;
        uploadFeedback.textContent = msg;
        setTimeout(() => {
            uploadFeedback.className = "feedback-msg";
        }, 5000);
    }

    // -----------------------------------------------------------------------
    // 4. File Dropzone & UI Utilities
    // -----------------------------------------------------------------------

    noticeFileInput.addEventListener("change", () => {
        if (noticeFileInput.files.length > 0) {
            dropzoneLabel.textContent = noticeFileInput.files[0].name;
        }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("drag-over");
            if (eventName === "drop" && e.dataTransfer.files.length > 0) {
                noticeFileInput.files = e.dataTransfer.files;
                dropzoneLabel.textContent = e.dataTransfer.files[0].name;
            }
        });
    });

    // Auto-resize input textarea as user types
    queryInput.addEventListener("input", autoResizeTextarea);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    function autoResizeTextarea() {
        queryInput.style.height = "auto";
        queryInput.style.height = `${Math.min(queryInput.scrollHeight, 140)}px`;
    }

    // Quick prompt chips
    document.querySelectorAll(".prompt-chip").forEach((btn) => {
        btn.addEventListener("click", () => {
            queryInput.value = btn.dataset.query;
            autoResizeTextarea();
            chatForm.dispatchEvent(new Event("submit"));
        });
    });

    // Reset Chat session
    newChatBtn.addEventListener("click", () => {
        conversationId = generateUUID();
        localStorage.setItem("campus_chat_convo_id", conversationId);
        chatFlow.innerHTML = "";
        if (welcomeScreen) welcomeScreen.style.display = "flex";
        queryInput.value = "";
        queryInput.focus();
    });

    // Sidebar Toggling for Mobile
    menuToggleBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
    closeSidebarBtn.addEventListener("click", () => sidebar.classList.remove("open"));
    refreshDocsBtn.addEventListener("click", loadIndexedDocuments);

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function refreshLucideIcons() {
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    function escapeHTML(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function generateUUID() {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = Math.random() * 16 | 0;
            const v = c === "x" ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Initial load
    loadIndexedDocuments();
});
