"""
main.py

FastAPI application for the Agentic AI Customer Support Assistant.

Key upgrades over the original RAG version:
  - POST /chat   — main agentic endpoint; accepts {message, session_id}
                   and runs the full ReAct agent loop (multi-turn, tool use,
                   self-correction, escalation).
  - POST /session/clear — resets a conversation session.
  - GET  /query + POST /query kept for backwards compatibility (single-turn,
    no session memory — legacy endpoint).
  - /health now reports agent status.
  - Startup hook: auto-seeds sample_policy.txt if no documents are indexed,
    so the bot is never empty after a cold start / Render redeploy.

The agent logic lives entirely in agent.py.
Document indexing / vector-search stays in rag_pipeline.py (ChromaDB).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import clear_session, run_agent
from rag_pipeline import (
    FRIENDLY_NOT_FOUND_MESSAGES,
    assess_relevance,
    build_prompt,
    check_endee_connection,
    ensure_index_exists,
    is_duplicate_document,
    list_documents,
    search,
    store_in_db,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})
MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SAMPLE_POLICY_PATH: Path = BASE_DIR / "sample_policy.txt"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic AI Customer Support Assistant",
    description=(
        "Multi-turn agentic customer support chatbot with tool use, "
        "self-correction, conversation memory, and human escalation."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


# ---------------------------------------------------------------------------
# Startup — initialise ChromaDB and auto-seed sample policy
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once when uvicorn starts the application.

    1. Ensures the ChromaDB collection is ready (creates it if it doesn't exist).
    2. If the knowledge base is completely empty, auto-indexes sample_policy.txt
       so the bot always has something to answer — even on a first deploy or
       after Render recycles the ephemeral disk.
    """
    logger.info("=== AI Customer Support Assistant v2.1.0 starting up ===")

    # Warm up ChromaDB (creates ./chroma_data/ if it doesn't exist)
    if not ensure_index_exists():
        logger.error("ChromaDB failed to initialise on startup!")
        return

    docs = list_documents()
    if not docs:
        logger.info("Knowledge base is empty — auto-seeding sample_policy.txt …")
        if SAMPLE_POLICY_PATH.exists():
            text = SAMPLE_POLICY_PATH.read_text(encoding="utf-8")
            if text.strip():
                result = store_in_db(text, source=SAMPLE_POLICY_PATH.name)
                if result["error"]:
                    logger.error("Auto-seed failed: %s", result["error"])
                else:
                    logger.info(
                        "Auto-seed complete: %d chunks from '%s'.",
                        result["chunks_indexed"],
                        SAMPLE_POLICY_PATH.name,
                    )
        else:
            logger.warning(
                "sample_policy.txt not found at '%s' — knowledge base will be empty.",
                SAMPLE_POLICY_PATH,
            )
    else:
        logger.info(
            "Knowledge base ready: %d document(s) already indexed.", len(docs)
        )

    logger.info("=== Startup complete. Agent is live and ready 24/7. ===")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(str(BASE_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    groq_configured = bool(GROQ_API_KEY)
    return {
        "embedder": True,
        "endee": check_endee_connection(),   # now checks ChromaDB
        "ollama": groq_configured,           # legacy field — true when Groq is configured
        "llm_provider": "groq",
        "llm_model": GROQ_MODEL,
        "groq_configured": groq_configured,
        "agent_mode": True,
        "vector_store": "chromadb",
        "availability": "24/7",
    }


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@app.get("/documents")
def documents() -> dict:
    return {"documents": list_documents()}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _validate_upload(filename: str | None, content: bytes) -> str | None:
    if not filename or not filename.strip():
        return "No file was provided."
    filename = filename.strip()
    if len(content) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return f"That file exceeds the {mb} MB limit."
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return f"'{ext or 'no extension'}' files aren't supported. Upload one of: {allowed}."
    if not content:
        return "That file is empty — there's nothing to index."
    if is_duplicate_document(filename):
        return (
            f"'{filename}' is already in the knowledge base. "
            "Rename the file or remove the existing copy first."
        )
    return None


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    err = _validate_upload(file.filename, content)
    if err:
        logger.warning("Rejected upload '%s': %s", file.filename, err)
        return {"error": err}
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "That file isn't valid UTF-8 text."}
    if not text.strip():
        return {"error": "That file contains only whitespace."}
    result = store_in_db(text, source=file.filename)
    if result["error"]:
        return {"error": result["error"]}
    if result["chunks_indexed"] == 0:
        return {"error": "No indexable text found in that file."}
    word = "chunk" if result["chunks_indexed"] == 1 else "chunks"
    logger.info("Upload: %s → %d %s.", file.filename, result["chunks_indexed"], word)
    return {
        "message": f"Indexed {result['chunks_indexed']} {word} from {file.filename}.",
        "filename": file.filename,
        "chunks_indexed": result["chunks_indexed"],
    }


# ---------------------------------------------------------------------------
# ── AGENTIC CHAT endpoint (new primary endpoint) ──────────────────────────
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""   # empty string → auto-generate a new session


@app.post("/chat")
def chat(body: ChatRequest) -> dict:
    """
    Primary agentic chat endpoint.

    - Accepts a user message and an optional session_id.
    - Runs the ReAct agent loop: the LLM autonomously decides which tools
      to call, can search multiple times, self-correct, and escalate.
    - Returns the final answer, the agent's tool-use trace, and session_id.

    Body:  {"message": "...", "session_id": "optional-uuid"}
    Response:
        {
          "answer":     str,
          "error":      str | null,
          "escalated":  bool,
          "trace":      [{tool, args, result, iteration}, ...],
          "timing":     {"total_ms": int, "iterations": int},
          "session_id": str
        }
    """
    session_id = body.session_id.strip() or str(uuid.uuid4())
    result = run_agent(body.message, session_id)
    return result


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class SessionClearRequest(BaseModel):
    session_id: str


@app.post("/session/clear")
def session_clear(body: SessionClearRequest) -> dict:
    """Clears conversation history for the given session (new conversation)."""
    clear_session(body.session_id)
    return {"cleared": True, "session_id": body.session_id}


# ---------------------------------------------------------------------------
# Legacy single-turn query endpoints (kept for backwards compatibility)
# These do NOT use the agent loop or session memory.
# ---------------------------------------------------------------------------


def _call_groq_simple(prompt: str) -> tuple[str | None, str | None]:
    """Simple one-shot Groq call used by the legacy /query endpoint."""
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY is not set."
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": False,
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to Groq API."
    except requests.exceptions.Timeout:
        return None, "Groq API timed out."
    except requests.exceptions.RequestException as exc:
        return None, f"Unexpected error: {exc}"
    if resp.status_code == 401:
        return None, "Invalid GROQ_API_KEY."
    if resp.status_code == 429:
        return None, "Groq rate limit reached. Please wait and retry."
    if resp.status_code != 200:
        return None, f"Groq error (HTTP {resp.status_code}): {resp.text[:200]}"
    try:
        return resp.json()["choices"][0]["message"]["content"].strip(), None
    except (KeyError, IndexError, ValueError) as exc:
        return None, f"Could not parse Groq response: {exc}"


def _run_query(q: str) -> dict:
    """Legacy single-turn RAG pipeline (no agent loop, no memory)."""
    if not q or not q.strip():
        return {
            "answer": "Please type a question so I can help you.",
            "relevant": False,
            "matches": [],
            "timing": {"retrieval_ms": 0, "generation_ms": 0},
            "error": None,
        }
    t0 = time.perf_counter()
    result = search(q.strip(), top_k=3)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    if result["error"]:
        return {
            "answer": None, "relevant": False, "matches": [],
            "timing": {"retrieval_ms": retrieval_ms, "generation_ms": 0},
            "error": f"Retrieval failed: {result['error']}",
        }
    matches = result["matches"]
    is_relevant, reason = assess_relevance(matches)
    if not is_relevant:
        return {
            "answer": FRIENDLY_NOT_FOUND_MESSAGES[reason],
            "relevant": False, "matches": matches,
            "timing": {"retrieval_ms": retrieval_ms, "generation_ms": 0},
            "error": None,
        }
    prompt = build_prompt(q.strip(), [m["text"] for m in matches])
    t1 = time.perf_counter()
    answer, error = _call_groq_simple(prompt)
    generation_ms = int((time.perf_counter() - t1) * 1000)
    return {
        "answer": answer, "relevant": True, "matches": matches,
        "timing": {"retrieval_ms": retrieval_ms, "generation_ms": generation_ms},
        "error": error,
    }


@app.get("/query")
def query_get(q: str) -> dict:
    """Legacy GET endpoint — no agent loop, no memory."""
    return _run_query(q)


class QueryBody(BaseModel):
    q: str


@app.post("/query")
def query_post(body: QueryBody) -> dict:
    """Legacy POST endpoint — no agent loop, no memory."""
    return _run_query(body.q)
