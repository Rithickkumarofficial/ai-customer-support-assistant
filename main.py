"""
main.py

FastAPI application for the AI Customer Support Assistant.

Route handlers stay thin: they validate input, delegate domain work to
rag_pipeline, call the Groq LLM API for generation, and shape the JSON
response.

All existing endpoints (/, /health, /documents, /upload, /query) are
preserved with the same paths and response fields so the frontend keeps
working without changes.  New additions:
  - POST /query   accepts a JSON body {"q": "…"} in addition to the
                  existing GET /query?q=… so long questions aren't
                  truncated by URL length limits.
  - /health now reports a "llm_provider" and "llm_model" field.

LLM backend: Groq Cloud (free tier, no credit card required).
  Get your free API key at https://console.groq.com
  Set it as GROQ_API_KEY in a .env file or environment variable.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_pipeline import (
    FRIENDLY_NOT_FOUND_MESSAGES,
    assess_relevance,
    build_prompt,
    check_endee_connection,
    is_duplicate_document,
    list_documents,
    search,
    store_in_db,
)

# Load .env file if present (GROQ_API_KEY, RELEVANCE_THRESHOLD, etc.)
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

# Explicit extension allowlist gives a clearer rejection message than a
# bare UnicodeDecodeError when someone uploads a PDF, etc.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})

# 5 MB is generous for .txt/.md policy documents.
MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024

# ---------------------------------------------------------------------------
# Groq LLM configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"

# llama-3.3-70b-versatile — best free Groq model for customer support tasks.
# Swap to "llama3-8b-8192" if you hit rate limits and need a faster/lighter model.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Customer Support Assistant",
    description=(
        "Retrieval-augmented customer support chatbot that answers only from "
        "your own uploaded policy documents."
    ),
    version="1.0.0",
)

# Wide-open CORS is fine for local / demo use.
# Tighten allow_origins to your real domain before going to production.
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
# Static frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serves the single-page chat UI (index.html)."""
    return FileResponse(str(BASE_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """
    Live connectivity status for each backend service the UI depends on.

    embedder — always True: SentenceTransformer loads in-process at startup.
    endee    — checked via HTTP ping to :8080.
    ollama   — always False now (replaced by Groq); kept in response so the
               existing frontend field names don't break.
    groq     — True if GROQ_API_KEY is set (we don't ping Groq on every
               health poll to avoid burning free-tier rate limit quota).
    """
    groq_configured = bool(GROQ_API_KEY)

    return {
        "embedder": True,
        "endee": check_endee_connection(),
        # Legacy field — kept so the frontend status panel still shows three
        # rows.  We repurpose it to reflect the Groq API key status.
        "ollama": groq_configured,
        "llm_provider": "groq",
        "llm_model": GROQ_MODEL,
        "groq_configured": groq_configured,
    }


# ---------------------------------------------------------------------------
# Document listing
# ---------------------------------------------------------------------------


@app.get("/documents")
def documents() -> dict:
    """Returns all indexed documents, most recently uploaded first."""
    return {"documents": list_documents()}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _validate_upload(filename: str | None, content: bytes) -> str | None:
    """
    Runs every upload pre-condition check in priority order.
    Returns a human-readable error string on failure, or None if OK.
    """
    if not filename or not filename.strip():
        return "No file was provided."

    filename = filename.strip()

    if len(content) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return f"That file exceeds the {mb} MB limit. Please split it into smaller documents."

    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return (
            f"'{ext or 'no extension'}' files aren't supported. "
            f"Please upload one of: {allowed}."
        )

    if not content:
        return "That file is empty — there's nothing to index."

    if is_duplicate_document(filename):
        return (
            f"'{filename}' is already in the knowledge base. "
            "Rename the file or remove the existing copy before uploading again."
        )

    return None


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """
    Accepts a .txt or .md policy document, chunks + embeds it with
    SentenceTransformers, and inserts it into the Endee vector database.
    """
    content = await file.read()

    validation_error = _validate_upload(file.filename, content)
    if validation_error:
        logger.warning("Rejected upload '%s': %s", file.filename, validation_error)
        return {"error": validation_error}

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "That file isn't valid UTF-8 text. Please save it as UTF-8 and try again."}

    if not text.strip():
        return {"error": "That file contains only whitespace — there's nothing to index."}

    result = store_in_db(text, source=file.filename)
    if result["error"]:
        return {"error": result["error"]}
    if result["chunks_indexed"] == 0:
        return {"error": "No indexable text was found in that file."}

    chunk_word = "chunk" if result["chunks_indexed"] == 1 else "chunks"
    logger.info("Upload accepted: %s → %d %s.", file.filename, result["chunks_indexed"], chunk_word)
    return {
        "message": f"Indexed {result['chunks_indexed']} {chunk_word} from {file.filename}.",
        "filename": file.filename,
        "chunks_indexed": result["chunks_indexed"],
    }


# ---------------------------------------------------------------------------
# Groq LLM generation
# ---------------------------------------------------------------------------


def _call_groq(prompt: str) -> tuple[str | None, str | None]:
    """
    Sends the assembled prompt to the Groq Chat Completions API.

    Uses the OpenAI-compatible /v1/chat/completions endpoint so the
    request shape is identical to the OpenAI SDK — easy to swap later.

    Returns (answer, error) — exactly one is always None.
    """
    if not GROQ_API_KEY:
        return None, (
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # The system prompt is already baked into the `prompt` string by
    # build_prompt().  We split it back into system / user messages here
    # so Groq's chat format gets the best possible instruction following.
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.3,      # low temperature → more factual, less creative
        "max_tokens": 1024,
        "stream": False,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to Groq API. Check your internet connection."
    except requests.exceptions.Timeout:
        return None, "Groq API took too long to respond. Please try again in a moment."
    except requests.exceptions.RequestException as exc:
        return None, f"Unexpected error contacting Groq: {exc}"

    if response.status_code == 401:
        return None, "Invalid GROQ_API_KEY. Check your key at https://console.groq.com"
    if response.status_code == 429:
        return None, "Groq rate limit reached. Please wait a moment and try again."
    if response.status_code != 200:
        snippet = response.text[:200]
        return None, f"Groq returned an error (HTTP {response.status_code}): {snippet}"

    try:
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer or "No answer received.", None
    except (KeyError, IndexError, ValueError) as exc:
        return None, f"Could not parse Groq response: {exc}"


# ---------------------------------------------------------------------------
# Query — shared handler
# ---------------------------------------------------------------------------


def _run_query(q: str) -> dict:
    """
    Core RAG pipeline:

    1. Embed question → retrieve top-k passages from Endee.
    2. Corrective-RAG gate: low relevance → friendly message, skip LLM.
    3. Build prompt → call Groq → return grounded answer + metadata.
    """
    if not q or not q.strip():
        return {
            "answer": "Please type a question so I can help you.",
            "relevant": False,
            "matches": [],
            "timing": {"retrieval_ms": 0, "generation_ms": 0},
            "error": None,
        }

    # ── Retrieval ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result = search(q.strip(), top_k=3)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    if result["error"]:
        logger.error("Retrieval failed for query '%s': %s", q, result["error"])
        return {
            "answer": None,
            "relevant": False,
            "matches": [],
            "timing": {"retrieval_ms": retrieval_ms, "generation_ms": 0},
            "error": f"Retrieval failed: {result['error']}",
        }

    matches = result["matches"]

    # ── Corrective-RAG gate ────────────────────────────────────────────────
    is_relevant, reason = assess_relevance(matches)
    if not is_relevant:
        logger.info("Query '%s' short-circuited (reason=%s) — skipping LLM call.", q, reason)
        return {
            "answer": FRIENDLY_NOT_FOUND_MESSAGES[reason],
            "relevant": False,
            "matches": matches,
            "timing": {"retrieval_ms": retrieval_ms, "generation_ms": 0},
            "error": None,
        }

    # ── Generation ─────────────────────────────────────────────────────────
    prompt = build_prompt(q.strip(), [m["text"] for m in matches])

    t1 = time.perf_counter()
    answer, error = _call_groq(prompt)
    generation_ms = int((time.perf_counter() - t1) * 1000)

    if error:
        logger.error("Generation failed for query '%s': %s", q, error)

    return {
        "answer": answer,
        "relevant": True,
        "matches": matches,
        "timing": {"retrieval_ms": retrieval_ms, "generation_ms": generation_ms},
        "error": error,
    }


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------


@app.get("/query")
def query_get(q: str) -> dict:
    """GET /query?q=<question> — original endpoint kept for compatibility."""
    return _run_query(q)


class QueryBody(BaseModel):
    q: str


@app.post("/query")
def query_post(body: QueryBody) -> dict:
    """POST /query  body: {"q": "question"} — preferred for long questions."""
    return _run_query(body.q)
