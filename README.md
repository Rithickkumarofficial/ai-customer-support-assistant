# AI Customer Support Assistant

> A retrieval-augmented, hallucination-resistant support chatbot that answers only from your own policy documents — and gives an honest "I don't know" when it can't find the answer.

---

## Problem Statement

Generic LLM chatbots confidently fabricate answers to support questions: wrong refund windows, invented warranty terms, made-up shipping fees. For customer support, a wrong answer is often worse than no answer — it erodes trust and creates legal exposure.

This project solves that by:

1. **Constraining the LLM** to answer only from uploaded company documents, never from its pre-trained knowledge.
2. **Adding a Corrective RAG gate** that measures retrieval confidence before generation. If the retrieved passages aren't convincing enough, Ollama is never called — the customer sees a warm escalation message instead.
3. **Full transparency** — every answer includes the source document, the exact passage retrieved, and its cosine similarity score.

---

## Technology Stack

| Layer             | Technology                                                      |
|-------------------|-----------------------------------------------------------------|
| Backend framework | **FastAPI** 0.111                                               |
| Vector database   | **Endee** (local HTTP API, port 8080)                           |
| Embeddings        | **Sentence-Transformers** `all-MiniLM-L6-v2` (384-d, local)    |
| LLM               | **Groq Cloud API** — `llama-3.3-70b-versatile` (free tier)     |
| Frontend          | HTML5, CSS3, vanilla JavaScript — zero build step               |
| Fonts             | Plus Jakarta Sans, IBM Plex Mono (Google Fonts)                 |

---

## Architecture

```
┌─────────────┐   POST /upload   ┌──────────────────┐
│  Browser    │ ───────────────▶ │  FastAPI          │
│  (Chat UI)  │                  │  main.py          │
│             │                  └────────┬──────────┘
│             │                           │ chunk + embed
│             │                  ┌────────▼──────────┐
│             │                  │  SentenceTransform│
│             │                  │  ers (in-process) │
│             │                  └────────┬──────────┘
│             │   GET /query              │ 384-d vectors
│             │ ───────────────▶          │ insert / search
│             │                  ┌────────▼──────────┐
│             │                  │  Endee vector DB  │
│             │                  │  (port 8080)      │
│             │                  └────────┬──────────┘
│             │                           │ top-k matches + scores
│             │                  ┌────────▼──────────┐
│             │                  │  Corrective RAG   │
│             │                  │  relevance gate   │
│             │                  └──────┬──────┬─────┘
│             │      low relevance ◀────┘      │ score ≥ threshold
│             │      (friendly msg,             │
│             │       LLM skipped)     ┌────────▼──────────┐
│             │                        │  Ollama Llama 3   │
│             │◀───────────────────────│  (port 11434)     │
│             │  answer + sources      └───────────────────┘
└─────────────┘
```

---

## Workflow

1. **Upload** — A `.txt` or `.md` policy document is validated (no empty files, no duplicates, no unsupported types, 5 MB ceiling), then chunked into ~3-sentence passages with one-sentence overlap between adjacent chunks, embedded with `all-MiniLM-L6-v2`, and inserted into Endee. A local JSON manifest tracks what's been indexed.

2. **Ask** — A customer types a question in the chat UI.

3. **Retrieve** — The question is embedded and Endee returns the `top_k=3` most similar passages, each with a cosine similarity score.

4. **Corrective RAG gate** — The highest similarity score is compared to a configurable threshold (default `0.40`). If it's below the threshold, or too few results came back, the pipeline short-circuits with a friendly message and **Ollama is never called**. No weak context, no hallucination.

5. **Generate** — If the gate passes, the passages are assembled into a system prompt that explicitly instructs the LLM to answer only from the provided context, never invent facts, and escalate when it can't help. The prompt is sent to Ollama.

6. **Respond** — The answer, retrieved sources (document name + passage text + similarity score), and retrieval/generation timing are returned and rendered in the chat UI behind a collapsible "Sources" toggle.

---

## Features

- **ChatGPT-style chat UI** with customer and agent bubbles, animated typing indicator, and auto-growing composer.
- **Corrective RAG** — retrieval quality is checked before generation; weak or sparse matches return a warm "I don't have that" message instead of a hallucinated answer.
- **Sentence-overlap chunking** — adjacent chunks share one overlapping sentence, preventing context loss at document boundaries.
- **Source transparency** — expand "Sources" on any answer to see the document name, passage text, and similarity score (shown as a %).
- **Upload validation** — empty files, oversized files, duplicate filenames, and unsupported extensions are rejected with a clear, specific message.
- **Live service status** — sidebar shows online/offline/checking for Embedder, Knowledge base, and AI model, polling every 15 seconds.
- **POST /query endpoint** — accepts `{"q": "…"}` in the request body alongside the existing GET endpoint, useful for long questions.
- **Responsive layout** — collapsible sidebar drawer on mobile; full two-panel layout on desktop.
- **Structured logging** throughout (Python `logging`, not `print`), with consistent timestamps and level names for easy grepping.
- **Tunable relevance threshold** via environment variable — no code change required.
- **Tunable minimum match count** via environment variable — reject answers when too few passages were retrieved, even if the top score is fine.

---

## Installation

```bash
pip install -r requirements.txt
```

**Prerequisites** — Endee must be running before starting the server:

| Service | Default port | Setup |
|---------|-------------|-------|
| [Endee](https://endee.io) | `:8080` | Follow Endee's quickstart guide |
| [Groq API](https://console.groq.com) | Cloud (free) | Sign up → create free API key |

Groq runs in the cloud — no local GPU or Ollama installation needed.

## Configuration

Copy `.env.example` to `.env` and fill in your Groq API key:

```bash
cp .env.example .env
# Then edit .env and set GROQ_API_KEY=your_key_here
```

Get your free key (no credit card required) at **https://console.groq.com**.

---

## Running

Place `index.html`, `main.py`, `rag_pipeline.py`, and the `static/` directory in the same folder, then:

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

**Optional — tune the pipeline without changing code:**

```bash
# Groq model (default: llama-3.3-70b-versatile)
export GROQ_MODEL=llama3-8b-8192   # faster/lighter alternative

# Corrective RAG threshold (default: 0.40)
export RELEVANCE_THRESHOLD=0.45

# Minimum retrieved passages before calling the LLM (default: 1)
export MIN_MATCH_COUNT=2

# Remote Endee instance
export ENDEE_URL=http://192.168.1.10:8080

uvicorn main:app --reload --port 8000
```

All variables can also live in a `.env` file — `python-dotenv` loads them automatically.

---

## Deployment

For a small production or internal deployment:

1. **Reverse proxy** — Run `uvicorn main:app --host 0.0.0.0 --port 8000` behind Nginx or Caddy to handle TLS termination and rate limiting.
2. **Process supervisor** — Use a systemd unit, `pm2`, or Supervisor so the server restarts automatically on crash or reboot.
3. **CORS** — Replace `allow_origins=["*"]` in `main.py` with your actual frontend domain(s) before going live.
4. **Services** — Endee and Ollama should run as long-lived services. Move them to separate machines by exporting `ENDEE_URL` in `rag_pipeline.py` and updating `OLLAMA_HOST` in `main.py`.
5. **Model pre-warming** — `sentence-transformers` downloads model weights on first run (~90 MB). Pre-bake them into your container image, or ensure outbound internet access during the first boot.
6. **Upload ceiling** — The default `MAX_UPLOAD_BYTES` in `main.py` is 5 MB. Raise it there if your policy documents are larger.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All three status dots are red | Backend isn't running | `uvicorn main:app --reload --port 8000` |
| Knowledge base is offline | Endee isn't running | Start Endee on port 8080 |
| "AI model" dot is red | GROQ_API_KEY not set | Add key to `.env` — get it free at console.groq.com |
| "Invalid API key" error | Wrong key | Double-check key at https://console.groq.com |
| "Rate limit" error | Free tier limit hit | Wait ~60 seconds and retry |
| Upload returns "already indexed" | Duplicate filename | Rename file or delete manifest |
| Every answer says "I don't have that" | Threshold too high / no docs uploaded | Lower `RELEVANCE_THRESHOLD` or upload a relevant document |
| Slow first answer | Model cold-start on Groq | Normal on first request; subsequent calls are fast |

---

## Future Scope

| Area | Description |
|------|-------------|
| Multi-turn memory | Remember earlier turns in a session so follow-up questions work naturally |
| Streaming responses | Stream tokens from Ollama instead of waiting for the full answer |
| Document management UI | Admin panel to delete, re-index, or preview individual documents |
| Human handoff | Wire the "connect me to an agent" escalation path to a real ticketing system (e.g. Zendesk, Freshdesk) |
| Multi-language | Detect the customer's language and reply in kind |
| Richer file types | Ingest PDF and DOCX in addition to `.txt`/`.md` |
| Cross-encoder re-ranking | Add a re-ranking step between retrieval and generation for better precision |
| Evaluation harness | Automated faithfulness and correctness scoring against a golden QA set |
| Auth + multi-tenant | Per-company knowledge bases with API key gating |
