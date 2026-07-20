# Agentic AI Customer Support Assistant

> A multi-turn agentic AI chatbot that autonomously searches your policy documents, self-corrects when results are weak, escalates to a human agent when it can't help — available **24/7**, powered by Groq LLM and a ReAct agent loop.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Rithickkumarofficial/ai-customer-support-assistant)

---

## What makes it Agentic?

| Feature | Description |
|---|---|
| **Tool use** | LLM autonomously decides which tool to call: `search_docs`, `rephrase_and_retry`, `get_document_list`, `check_knowledge_base`, `get_current_time`, `escalate_to_human` |
| **Multi-turn memory** | Full conversation history kept per session — follow-up questions work naturally |
| **Self-correction** | If search results are weak, agent rephrases and retries automatically |
| **Escalation** | When no answer is found, agent calls `escalate_to_human` with a context summary |
| **ReAct loop** | Think → Act → Observe → repeat (up to 5 iterations per message) |
| **Agent trace** | Every tool call + result shown in collapsible UI panel |
| **24/7 availability** | Groq retry logic with exponential back-off on rate-limits; always responds |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI 0.111 |
| LLM | Groq Cloud — `llama-3.3-70b-versatile` (free tier) |
| Embeddings | **fastembed** `all-MiniLM-L6-v2` via ONNX Runtime (~80 MB RAM, no PyTorch) |
| Vector Store | **ChromaDB** (persistent, pure Python — no separate server needed) |
| Frontend | HTML5 + CSS3 + Vanilla JS |

---

## Deploy to Render (1-click)

### Step 1 — Fork or push to GitHub
Your repo: `https://github.com/Rithickkumarofficial/ai-customer-support-assistant`

### Step 2 — Create a Render account
Go to [render.com](https://render.com) and sign up for free.

### Step 3 — New Web Service from GitHub

1. Click **New → Web Service**
2. Connect your GitHub account
3. Select the repo `ai-customer-support-assistant`
4. Render auto-detects `render.yaml` — settings are pre-filled:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`

### Step 4 — Set environment variable

In the Render dashboard under **Environment**:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_your_key_here` (get free at [console.groq.com](https://console.groq.com)) |

### Step 5 — Deploy
Click **Create Web Service**. Render builds and deploys automatically.
Your app will be live at: `https://ai-customer-support-assistant.onrender.com`

> **Note on memory**: The app uses ~175 MB RAM (well within Render's 512 MB free limit).
> `sample_policy.txt` is auto-indexed on every cold start so the bot always has answers.

---

## Run locally

```bash
# 1. Clone
git clone https://github.com/Rithickkumarofficial/ai-customer-support-assistant.git
cd ai-customer-support-assistant

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_key

# 5. Start server
uvicorn main:app --reload --port 8000

# 6. Open browser
open http://localhost:8000
```

---

## How it works

```
User message
     │
     ▼
┌──────────────────────────────────────┐
│          ReAct Agent Loop            │
│                                      │
│  Think: What tool do I need?         │
│     │                                │
│     ▼                                │
│  Act: Call tool                      │
│  • search_docs(query)                │
│  • rephrase_and_retry(query)         │
│  • get_document_list()               │
│  • check_knowledge_base()            │
│  • get_current_time()                │
│  • escalate_to_human(reason)         │
│     │                                │
│     ▼                                │
│  Observe: Get tool result            │
│     │                                │
│     └─► Loop up to 5x ──────────►   │
│                                      │
│  Final: Generate answer              │
└──────────────────────────────────────┘
     │
     ▼
Response + Agent trace shown in UI
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | **Main agentic endpoint** — `{message, session_id}` |
| `POST` | `/session/clear` | Reset conversation memory |
| `POST` | `/upload` | Upload `.txt` or `.md` policy document |
| `GET` | `/documents` | List indexed documents |
| `GET` | `/health` | Service status (embedder, vector store, LLM) |
| `GET` | `/` | Chat UI |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "GROQ_API_KEY is not set" | Add key to `.env` or Render environment variables |
| Slow first response | Normal — ONNX embedding model loads on first request (~5s) |
| "Agent escalated" on every question | Upload a relevant policy document first |
| Port already in use | `lsof -ti:8000 \| xargs kill -9` |
| Memory exceeded on Render | Ensure `--workers 1` is in start command (already in render.yaml) |
