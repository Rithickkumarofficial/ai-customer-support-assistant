(() => {
  "use strict";

  // -----------------------------------------------------------------------
  // DOM refs
  // -----------------------------------------------------------------------
  const app             = document.getElementById("app");
  const menuToggle      = document.getElementById("menu-toggle");
  const sidebarBackdrop = document.getElementById("sidebar-backdrop");

  const dropzone          = document.getElementById("dropzone");
  const fileInput         = document.getElementById("file-input");
  const uploadProgress    = document.getElementById("upload-progress");
  const uploadProgressBar = document.getElementById("upload-progress-bar");
  const uploadFeedback    = document.getElementById("upload-feedback");
  const docList           = document.getElementById("doc-list");
  const docEmpty          = document.getElementById("doc-empty");
  const docCount          = document.getElementById("doc-count");

  const statusList    = document.getElementById("status-list");
  const chatScroll    = document.getElementById("chat-scroll");
  const welcome       = document.getElementById("welcome");
  const newChatBtn    = document.getElementById("new-chat-btn");
  const composer      = document.getElementById("composer");
  const composerInput = document.getElementById("composer-input");
  const composerSend  = document.getElementById("composer-send");

  // -----------------------------------------------------------------------
  // Session state — persists for the lifetime of the page
  // -----------------------------------------------------------------------
  let sessionId  = "";   // assigned after first /chat response
  let isSending  = false;

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatAnswer(text) {
    const escaped = escapeHtml(text);
    const withCode = escaped
      .replace(/```(?:\w*)\n?([\s\S]*?)```/g, (_, code) =>
        `<pre><code>${code.trim()}</code></pre>`
      )
      .replace(/`([^`]+)`/g, "<code>$1</code>");
    return withCode
      .split(/\n\s*\n/)
      .map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`)
      .join("");
  }

  function formatMs(ms) {
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
  }

  function timeAgo(isoString) {
    const then = new Date(isoString).getTime();
    if (Number.isNaN(then)) return "";
    const diffSec = Math.max(0, Math.floor((Date.now() - then) / 1000));
    if (diffSec < 5)   return "just now";
    if (diffSec < 60)  return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60)  return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr  < 24)  return `${diffHr}h ago`;
    return `${Math.floor(diffHr / 24)}d ago`;
  }

  function scoreInfo(score) {
    if (typeof score !== "number") return { label: "—", cls: "score-low", pct: 0 };
    const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
    const cls = score >= 0.65 ? "score-high" : score >= 0.45 ? "score-mid" : "score-low";
    return { label: `${pct}%`, cls, pct };
  }

  // -----------------------------------------------------------------------
  // Mobile sidebar
  // -----------------------------------------------------------------------

  function setSidebarOpen(open) {
    app.classList.toggle("sidebar-open", open);
    menuToggle.setAttribute("aria-expanded", String(open));
  }
  menuToggle.addEventListener("click", () =>
    setSidebarOpen(!app.classList.contains("sidebar-open"))
  );
  sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false));

  // -----------------------------------------------------------------------
  // Health polling
  // -----------------------------------------------------------------------

  function setStatus(service, state) {
    const item = statusList.querySelector(`[data-service="${service}"]`);
    if (!item) return;
    item.classList.remove("is-online", "is-offline", "is-checking");
    item.classList.add(state);
    const badge = item.querySelector(".status-badge");
    if (badge) {
      badge.textContent =
        state === "is-online"  ? "online"  :
        state === "is-offline" ? "offline" : "…";
    }
  }

  async function refreshHealth() {
    ["embedder", "endee", "ollama"].forEach(s => setStatus(s, "is-checking"));
    try {
      const res  = await fetch("/health");
      const data = await res.json();
      setStatus("embedder", data.embedder ? "is-online" : "is-offline");
      setStatus("endee",    data.endee    ? "is-online" : "is-offline");
      setStatus("ollama",   data.ollama   ? "is-online" : "is-offline");
    } catch {
      ["embedder", "endee", "ollama"].forEach(s => setStatus(s, "is-offline"));
    }
  }

  // -----------------------------------------------------------------------
  // Document list
  // -----------------------------------------------------------------------

  function renderDocuments(docs) {
    docCount.textContent = String(docs.length);
    docList.querySelectorAll(".doc-item").forEach(el => el.remove());
    docEmpty.style.display = docs.length ? "none" : "block";
    for (const doc of docs) {
      const li = document.createElement("li");
      li.className = "doc-item";
      li.innerHTML = `
        <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
        <div class="doc-meta">
          <span class="doc-chunks">${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}</span>
          <span>${escapeHtml(timeAgo(doc.uploaded_at))}</span>
        </div>`;
      docList.appendChild(li);
    }
  }

  async function refreshDocuments() {
    try {
      const res  = await fetch("/documents");
      const data = await res.json();
      renderDocuments(data.documents || []);
    } catch { /* backend down — status cluster already shows it */ }
  }

  // -----------------------------------------------------------------------
  // File upload
  // -----------------------------------------------------------------------

  function setUploadFeedback(message, isError) {
    uploadFeedback.textContent = message;
    uploadFeedback.classList.toggle("is-error", Boolean(isError));
  }

  function setUploadProgress(active) {
    uploadProgress.hidden = !active;
    uploadProgressBar.style.width = active ? "70%" : "0%";
  }

  async function handleFile(file) {
    if (!file) return;
    if (!/\.(txt|md)$/i.test(file.name)) {
      setUploadFeedback("Only .txt or .md files are supported.", true);
      return;
    }
    setUploadFeedback(`Indexing ${file.name}…`, false);
    setUploadProgress(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res  = await fetch("/upload", { method: "POST", body: formData });
      const data = await res.json();
      uploadProgressBar.style.width = "100%";
      if (data.error) {
        setUploadFeedback(data.error, true);
      } else {
        setUploadFeedback(`✓ ${data.message}`, false);
        refreshDocuments();
      }
    } catch {
      setUploadFeedback("Couldn't reach the backend. Is it running?", true);
      refreshHealth();
    } finally {
      fileInput.value = "";
      setTimeout(() => setUploadProgress(false), 500);
    }
  }

  dropzone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", e => handleFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach(evt =>
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "dragend", "drop"].forEach(evt =>
    dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover"))
  );
  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });

  // -----------------------------------------------------------------------
  // Chat rendering
  // -----------------------------------------------------------------------

  function scrollToBottom() {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  function addCustomerMessage(text) {
    welcome.style.display = "none";
    const msg = document.createElement("div");
    msg.className = "msg msg-customer";
    msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatScroll.appendChild(msg);
    scrollToBottom();
  }

  function addPendingAgentMessage() {
    const msg = document.createElement("div");
    msg.className = "msg msg-agent msg-pending";
    msg.innerHTML = `
      <div class="avatar" aria-hidden="true">
        <svg viewBox="0 0 32 32">
          <rect width="32" height="32" rx="9" fill="var(--brand)"/>
          <path d="M8 12.5c0-2.2 1.8-4 4-4h8c2.2 0 4 1.8 4 4v5c0 2.2-1.8 4-4 4h-6l-3.5 3v-3H12c-2.2 0-4-1.8-4-4v-5z" fill="white"/>
          <path d="M12.5 15.2l2 2 4-4.2" stroke="var(--brand)" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="msg-body">
        <div class="bubble typing-bubble" aria-label="Agent is thinking">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>`;
    chatScroll.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  /** Renders the agent's tool-use trace as a collapsible panel */
  function renderTrace(trace) {
    if (!trace || trace.length === 0) return "";

    const TOOL_ICONS = {
      search_docs:        "🔍",
      rephrase_and_retry: "🔄",
      get_document_list:  "📂",
      escalate_to_human:  "🙋",
    };

    const steps = trace.map((step, i) => {
      const icon = TOOL_ICONS[step.tool] || "🔧";
      const argsStr = escapeHtml(JSON.stringify(step.args, null, 2));

      // For search tools, render match summaries
      let resultHtml = "";
      const r = step.result;
      if ((step.tool === "search_docs" || step.tool === "rephrase_and_retry") && r.matches) {
        if (r.matches.length === 0) {
          resultHtml = `<p class="trace-no-results">No matches found.</p>`;
        } else {
          resultHtml = r.matches.map(m => {
            const { label, cls } = scoreInfo(m.score);
            return `
              <div class="trace-match">
                <span class="match-rank">#${m.rank}</span>
                <span class="match-source">${escapeHtml(m.source || "unknown")}</span>
                <span class="match-score-badge ${cls}">${label}</span>
                <p class="match-text">${escapeHtml((m.text || "").slice(0, 140))}…</p>
              </div>`;
          }).join("");
        }
      } else if (step.tool === "get_document_list" && r.documents) {
        resultHtml = `<p class="trace-note">${r.total} document(s) indexed.</p>`;
      } else if (step.tool === "escalate_to_human") {
        resultHtml = `<p class="trace-note escalation">🙋 Escalated: ${escapeHtml(r.reason || "")}</p>`;
      } else if (r.error) {
        resultHtml = `<p class="trace-note error">Error: ${escapeHtml(r.error)}</p>`;
      }

      return `
        <div class="trace-step">
          <div class="trace-step-header">
            <span class="trace-iter">Step ${i + 1}</span>
            <span class="trace-tool-name">${icon} ${escapeHtml(step.tool)}</span>
          </div>
          <pre class="trace-args">${argsStr}</pre>
          <div class="trace-result">${resultHtml}</div>
        </div>`;
    }).join("");

    return `
      <div class="msg-meta">
        <button class="sources-toggle trace-toggle" type="button" aria-expanded="false">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
          Agent trace (${trace.length} step${trace.length !== 1 ? "s" : ""})
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>
        </button>
      </div>
      <div class="sources-panel trace-panel" hidden>${steps}</div>`;
  }

  function resolveAgentMessage(pendingEl, data) {
    const hasHardError  = Boolean(data.error);
    const isEscalated   = Boolean(data.escalated);

    let bubbleClass = "bubble";
    if (hasHardError) bubbleClass += " error-bubble";
    if (isEscalated)  bubbleClass += " escalated-bubble";

    const answerText = data.error || data.answer || "No answer received.";

    const timing = data.timing;
    const timingLabel = timing
      ? `${formatMs(timing.total_ms)} total &nbsp;·&nbsp; ${timing.iterations} agent step${timing.iterations !== 1 ? "s" : ""}`
      : "";

    const traceHtml  = renderTrace(data.trace);

    let metaHtml = "";
    if (!hasHardError && timingLabel) {
      metaHtml = `<div class="msg-timing">${timingLabel}</div>`;
    }

    pendingEl.classList.remove("msg-pending");
    pendingEl.querySelector(".msg-body").innerHTML = `
      <div class="${bubbleClass}">
        <div class="answer-text">${formatAnswer(answerText)}</div>
      </div>
      ${metaHtml}
      ${traceHtml}`;

    // Wire trace toggle
    const toggle = pendingEl.querySelector(".trace-toggle");
    if (toggle) {
      const panel = pendingEl.querySelector(".trace-panel");
      toggle.addEventListener("click", () => {
        const expanded = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", String(!expanded));
        panel.hidden = expanded;
        if (!expanded) scrollToBottom();
      });
    }

    scrollToBottom();
  }

  function resolveAgentMessageWithNetworkError(pendingEl) {
    pendingEl.classList.remove("msg-pending");
    pendingEl.querySelector(".msg-body").innerHTML = `
      <div class="bubble error-bubble">
        Couldn't reach the backend. Make sure the FastAPI server is running.
      </div>`;
    scrollToBottom();
  }

  // -----------------------------------------------------------------------
  // Send message — calls agentic /chat endpoint
  // -----------------------------------------------------------------------

  async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed || isSending) return;

    isSending = true;
    composerSend.disabled = true;
    addCustomerMessage(trimmed);
    composerInput.value = "";
    composerInput.style.height = "auto";

    const pending = addPendingAgentMessage();

    try {
      const res  = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, session_id: sessionId }),
      });
      const data = await res.json();

      // Persist session_id for subsequent turns
      if (data.session_id) sessionId = data.session_id;

      resolveAgentMessage(pending, data);
    } catch {
      resolveAgentMessageWithNetworkError(pending);
      refreshHealth();
    } finally {
      isSending = false;
      composerSend.disabled = false;
      composerInput.focus();
    }
  }

  // -----------------------------------------------------------------------
  // Composer events
  // -----------------------------------------------------------------------

  composer.addEventListener("submit", e => { e.preventDefault(); sendMessage(composerInput.value); });
  composerInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(composerInput.value); }
  });
  composerInput.addEventListener("input", () => {
    composerInput.style.height = "auto";
    composerInput.style.height = `${Math.min(composerInput.scrollHeight, 140)}px`;
  });

  document.querySelectorAll(".suggestion").forEach(btn => {
    btn.addEventListener("click", () => {
      composerInput.value = btn.dataset.prompt;
      composerInput.focus();
      composerInput.dispatchEvent(new Event("input"));
    });
  });

  // New conversation — clears session on backend too
  newChatBtn.addEventListener("click", async () => {
    if (sessionId) {
      try {
        await fetch("/session/clear", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
      } catch { /* ignore — session will expire naturally */ }
    }
    sessionId = "";
    chatScroll.querySelectorAll(".msg").forEach(el => el.remove());
    welcome.style.display = "flex";
    composerInput.value = "";
    composerInput.style.height = "auto";
    composerInput.focus();
    setSidebarOpen(false);
  });

  // -----------------------------------------------------------------------
  // Init
  // -----------------------------------------------------------------------

  refreshHealth();
  refreshDocuments();
  setInterval(refreshHealth, 15_000);

})();
