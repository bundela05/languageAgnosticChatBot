/**
 * Language-Agnostic Campus Chatbot Frontend Engine
 * Coordinates UI interactions, SSE chat streaming, markdown rendering, and document indexing.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------------------------
    // 1. State Management & Initialization
    // ----------------------------------------------------------------------
    const STORAGE_KEY_CONVERSATION_ID = "campus_chatbot_conv_id";
    let conversationId = localStorage.getItem(STORAGE_KEY_CONVERSATION_ID);

    if (!conversationId) {
        conversationId = generateUUID();
        localStorage.setItem(STORAGE_KEY_CONVERSATION_ID, conversationId);
    }

    // DOM Elements
    const chatFlow = document.getElementById("chatFlow");
    const welcomeScreen = document.getElementById("welcomeScreen");
    const messagesContainer = document.getElementById("messagesContainer");
    const chatForm = document.getElementById("chatForm");
    const queryInput = document.getElementById("queryInput");
    const sendBtn = document.getElementById("sendBtn");
    const languageSelect = document.getElementById("languageSelect");
    const newChatBtn = document.getElementById("newChatBtn");
    const menuToggleBtn = document.getElementById("menuToggleBtn");
    const closeSidebarBtn = document.getElementById("closeSidebarBtn");
    const sidebar = document.getElementById("sidebar");

    // Document Management DOM Elements
    const uploadForm = document.getElementById("uploadForm");
    const noticeFileInput = document.getElementById("noticeFileInput");
    const dropzone = document.getElementById("dropzone");
    const dropzoneLabel = document.getElementById("dropzoneLabel");
    const uploadSubmitBtn = document.getElementById("uploadSubmitBtn");
    const uploadFeedback = document.getElementById("uploadFeedback");
    const indexedDocList = document.getElementById("indexedDocList");
    const refreshDocsBtn = document.getElementById("refreshDocsBtn");

    // Configure marked.js for sanitized and styled tables/lists
    if (window.marked) {
        marked.setOptions({
            gfm: true,
            breaks: true,
            sanitize: false
        });
    }

    // Render initial icons and fetch active documents
    refreshLucideIcons();
    fetchIndexedDocuments();

    // ----------------------------------------------------------------------
    // 2. Chat Interaction & Server-Sent Events (SSE) Streaming
    // ----------------------------------------------------------------------

    // Auto-expand textarea height as user types
    queryInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 140) + "px";
    });

    // Handle Enter key for submission (Shift+Enter for newline)
    queryInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Quick prompt chips
    document.querySelectorAll(".prompt-chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            if (query) {
                queryInput.value = query;
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    // Submit user message
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (!text) return;

        // Hide welcome screen on first question
        if (welcomeScreen) {
            welcomeScreen.style.display = "none";
        }

        // Render user message bubble
        appendUserMessage(text);
        queryInput.value = "";
        queryInput.style.height = "auto";
        sendBtn.disabled = true;

        // Render assistant placeholder bubble with typing animation
        const botBubbleContainer = createBotMessagePlaceholder();
        chatFlow.appendChild(botBubbleContainer);
        scrollToBottom();

        const selectedLang = languageSelect.value;

        try {
            await handleStreamingChat(text, selectedLang, botBubbleContainer);
        } catch (err) {
            console.warn("SSE stream failed, falling back to standard POST:", err);
            await handleStandardChat(text, selectedLang, botBubbleContainer);
        } finally {
            sendBtn.disabled = false;
            queryInput.focus();
            refreshLucideIcons();
        }
    });

    /**
     * Executes streaming inference via /api/chat/stream
     */
    async function handleStreamingChat(queryText, prefLang, container) {
        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: queryText,
                conversation_id: conversationId,
                preferred_language: prefLang
            })
        });

        if (!response.ok || !response.body) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        const contentDiv = container.querySelector(".bot-content");
        let accumulatedText = "";
        let metaData = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const dataStr = line.replace("data: ", "").trim();
                    if (dataStr === "[DONE]") {
                        break;
                    }

                    try {
                        const event = JSON.parse(dataStr);
                        if (event.type === "metadata") {
                            metaData = event;
                            renderBotMetadata(container, event);
                        } else if (event.type === "content") {
                            accumulatedText += event.delta;
                            contentDiv.innerHTML = window.marked
                                ? marked.parse(accumulatedText)
                                : escapeHtml(accumulatedText);
                            scrollToBottom();
                        }
                    } catch (pErr) {
                        // ignore malformed partial SSE chunks
                    }
                }
            }
        }

        // Render citations if available
        if (metaData && metaData.sources && metaData.sources.length > 0) {
            renderSources(container, metaData.sources);
        }
    }

    /**
     * Fallback standard POST to /api/chat
     */
    async function handleStandardChat(queryText, prefLang, container) {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: queryText,
                conversation_id: conversationId,
                preferred_language: prefLang
            })
        });

        const data = await response.json();
        const contentDiv = container.querySelector(".bot-content");

        if (response.ok) {
            contentDiv.innerHTML = window.marked
                ? marked.parse(data.response)
                : escapeHtml(data.response);
            renderBotMetadata(container, data);
            if (data.sources && data.sources.length > 0) {
                renderSources(container, data.sources);
            }
        } else {
            contentDiv.innerHTML = `<span style="color: var(--accent-rose);">Error: ${escapeHtml(data.detail || "Unable to process query.")}</span>`;
        }
        scrollToBottom();
    }

    // ----------------------------------------------------------------------
    // 3. UI Rendering Helpers
    // ----------------------------------------------------------------------

    function appendUserMessage(text) {
        const row = document.createElement("div");
        row.className = "message-row user";
        row.innerHTML = `
            <div class="message-body">
                <div class="message-bubble">${escapeHtml(text)}</div>
            </div>
        `;
        chatFlow.appendChild(row);
        scrollToBottom();
    }

    function createBotMessagePlaceholder() {
        const row = document.createElement("div");
        row.className = "message-row bot";
        row.innerHTML = `
            <div class="avatar bot-avatar">
                <i data-lucide="bot"></i>
            </div>
            <div class="message-body">
                <div class="message-bubble bot-content">
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
                <div class="bubble-meta" style="display: none;"></div>
                <div class="sources-container"></div>
            </div>
        `;
        return row;
    }

    function renderBotMetadata(container, data) {
        const metaDiv = container.querySelector(".bubble-meta");
        if (!metaDiv) return;

        const lang = data.detected_language || "en";
        const intent = data.intent || "campus_query";

        metaDiv.style.display = "flex";
        metaDiv.innerHTML = `
            <span class="lang-pill"><i data-lucide="globe" style="width:10px;height:10px;display:inline;"></i> ${lang}</span>
            <span class="intent-pill">${intent.replace("_", " ")}</span>
        `;
        refreshLucideIcons();
    }

    function renderSources(container, sources) {
        const sourcesContainer = container.querySelector(".sources-container");
        if (!sourcesContainer || sources.length === 0) return;

        let sourcesHtml = '<div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.3rem;">Grounded Sources:</div>';
        sources.forEach((s) => {
            const pageInfo = s.page_number ? ` (p. ${s.page_number})` : "";
            const scoreInfo = s.relevance_score ? ` [${Math.round(s.relevance_score * 100)}% match]` : "";
            sourcesHtml += `
                <div class="source-badge" title="${escapeHtml(s.chunk_text.slice(0, 150))}...">
                    <i data-lucide="file-check"></i>
                    <span>${escapeHtml(s.document_title)}${pageInfo}${scoreInfo}</span>
                </div>
            `;
        });

        sourcesContainer.innerHTML = sourcesHtml;
        refreshLucideIcons();
        scrollToBottom();
    }

    // ----------------------------------------------------------------------
    // 4. Document Upload & Repository Management
    // ----------------------------------------------------------------------

    // File drag & drop triggers
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("drag-over");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            noticeFileInput.files = files;
            dropzoneLabel.textContent = files[0].name;
        }
    });

    noticeFileInput.addEventListener("change", function () {
        if (this.files.length > 0) {
            dropzoneLabel.textContent = this.files[0].name;
        }
    });

    // Form submission for document indexing
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!noticeFileInput.files || noticeFileInput.files.length === 0) {
            showFeedback("Please select a valid PDF or TXT file first.", "error");
            return;
        }

        const file = noticeFileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);

        uploadSubmitBtn.disabled = true;
        uploadSubmitBtn.innerHTML = "<span>Indexing document...</span>";
        showFeedback("Processing layout and generating vector embeddings...", "success");

        try {
            const res = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            if (res.ok) {
                showFeedback(`Indexed ${data.filename} (${data.chunks_created} chunks)`, "success");
                uploadForm.reset();
                dropzoneLabel.textContent = "Click or drag PDF here";
                fetchIndexedDocuments();
            } else {
                showFeedback(`Upload failed: ${data.detail || "Server error"}`, "error");
            }
        } catch (err) {
            showFeedback(`Error uploading notice: ${err.message}`, "error");
        } finally {
            uploadSubmitBtn.disabled = false;
            uploadSubmitBtn.innerHTML = "<span>Index Document</span>";
        }
    });

    // Fetch and render indexed documents in drawer
    async function fetchIndexedDocuments() {
        try {
            const res = await fetch("/api/documents/list");
            if (!res.ok) throw new Error("Could not fetch documents");

            const docs = await res.json();
            indexedDocList.innerHTML = "";

            if (!docs || docs.length === 0) {
                indexedDocList.innerHTML = '<li class="empty-doc-item">No campus circulars indexed yet.</li>';
                return;
            }

            docs.forEach((doc) => {
                const li = document.createElement("li");
                li.className = "doc-item";
                li.innerHTML = `
                    <div class="doc-info">
                        <span class="doc-title" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
                        <span class="doc-meta">${doc.chunks_count} chunks • ${doc.category || "General"}</span>
                    </div>
                    <button class="doc-delete-btn" data-filename="${escapeHtml(doc.filename)}" title="Remove Notice">
                        <i data-lucide="trash-2"></i>
                    </button>
                `;

                li.querySelector(".doc-delete-btn").addEventListener("click", async function () {
                    const filename = this.getAttribute("data-filename");
                    if (confirm(`Remove '${filename}' and purge vector embeddings?`)) {
                        await deleteNotice(filename);
                    }
                });

                indexedDocList.appendChild(li);
            });
            refreshLucideIcons();
        } catch (err) {
            indexedDocList.innerHTML = '<li class="empty-doc-item">Unable to load notices.</li>';
        }
    }

    async function deleteNotice(filename) {
        try {
            const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
                method: "DELETE"
            });
            if (res.ok) {
                fetchIndexedDocuments();
            } else {
                alert("Failed to delete document.");
            }
        } catch (e) {
            alert(`Error deleting document: ${e.message}`);
        }
    }

    refreshDocsBtn.addEventListener("click", () => {
        fetchIndexedDocuments();
    });

    // ----------------------------------------------------------------------
    // 5. Sidebar Toggle & Session Reset
    // ----------------------------------------------------------------------
    menuToggleBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
    });

    closeSidebarBtn.addEventListener("click", () => {
        sidebar.classList.remove("open");
    });

    newChatBtn.addEventListener("click", () => {
        if (confirm("Start a new conversation session?")) {
            conversationId = generateUUID();
            localStorage.setItem(STORAGE_KEY_CONVERSATION_ID, conversationId);
            chatFlow.innerHTML = "";
            if (welcomeScreen) {
                welcomeScreen.style.display = "flex";
            }
        }
    });

    // ----------------------------------------------------------------------
    // 6. Utility Functions
    // ----------------------------------------------------------------------
    function generateUUID() {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
            const r = (Math.random() * 16) | 0;
            const v = c === "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function showFeedback(text, type) {
        uploadFeedback.textContent = text;
        uploadFeedback.className = `feedback-msg ${type}`;
        uploadFeedback.style.display = "block";
        setTimeout(() => {
            if (type === "success") {
                uploadFeedback.style.display = "none";
            }
        }, 5000);
    }

    function escapeHtml(string) {
        const div = document.createElement("div");
        div.textContent = string;
        return div.innerHTML;
    }

    function refreshLucideIcons() {
        if (window.lucide) {
            lucide.createIcons();
        }
    }
});
