"""
agent.py

Agentic AI core for the Customer Support Assistant.

Architecture: ReAct loop (Reason → Act → Observe → repeat).

The agent is given a set of tools and a conversation history. On each turn:
  1. The LLM reasons about what to do next (thinks).
  2. It emits a tool_call (acts).
  3. We execute the tool and return the observation.
  4. The LLM processes the observation and either calls another tool or
     produces a final answer.

This loop continues for up to MAX_ITERATIONS steps, then the agent is forced
to produce a final answer with whatever context it has gathered.

Tools available to the agent
─────────────────────────────
• search_docs(query)           — semantic search in the ChromaDB vector DB
• rephrase_and_retry(reason)   — self-correction: rephrase the user question
                                  and search again with a better query
• get_document_list()          — list all indexed policy documents
• check_knowledge_base()       — check if KB has any documents before searching
• get_current_time()           — return current UTC time (useful for 24/7 status)
• escalate_to_human(reason)    — generate a warm handoff message and stop

Robustness (24/7 operation)
────────────────────────────
• Groq rate-limit (HTTP 429) and server-error (HTTP 503) retries with
  exponential back-off — up to 3 retries per call.
• Session memory pruning on every request.

Memory
──────
Conversation sessions are held in a per-process in-memory dict keyed by
session_id (UUID).  Each session stores the full message history so the LLM
can understand follow-up questions ("what about the refund on that order?").

Sessions expire after SESSION_TTL_SECONDS of inactivity (default 30 min).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from rag_pipeline import list_documents, search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))  # 30 min

# Groq retry settings — handles 429 rate-limit and transient 5xx errors
_GROQ_MAX_RETRIES: int = 3
_GROQ_RETRY_BASE_DELAY: float = 2.0   # seconds; doubles on each retry

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}


def get_or_create_session(session_id: str) -> dict:
    """Returns existing session or creates a fresh one."""
    now = time.time()
    _prune_expired_sessions(now)

    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],       # list[{"role": str, "content": str | list}]
            "created_at": now,
            "last_active": now,
        }
        logger.info("New session created: %s", session_id)
    else:
        _sessions[session_id]["last_active"] = now

    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    """Deletes a session (used when the user starts a new conversation)."""
    _sessions.pop(session_id, None)
    logger.info("Session cleared: %s", session_id)


def _prune_expired_sessions(now: float) -> None:
    expired = [
        sid
        for sid, sess in _sessions.items()
        if now - sess["last_active"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]
        logger.debug("Session expired and pruned: %s", sid)


# ---------------------------------------------------------------------------
# Tool definitions (sent to Groq as function schemas)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the company's knowledge base (policy documents) for passages "
                "relevant to a customer question. Returns the top matching passages with "
                "similarity scores. Use this whenever the customer asks about policies, "
                "procedures, orders, returns, shipping, warranties, or any topic that "
                "might be covered by uploaded documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Rephrase the customer question as a "
                            "concise, keyword-rich statement for best retrieval results."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rephrase_and_retry",
            "description": (
                "If search_docs returned low-confidence or irrelevant results, call this "
                "tool to try a different search query. Provide a revised query that "
                "approaches the customer's question from a different angle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_query": {
                        "type": "string",
                        "description": "A rephrased or alternative search query to try.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief note on why you are rephrasing (for the trace log).",
                    },
                },
                "required": ["new_query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_list",
            "description": (
                "Returns the names and chunk counts of all documents currently indexed "
                "in the knowledge base. Useful when the customer asks what topics are "
                "covered, or when deciding whether a document relevant to their question "
                "has been uploaded."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_knowledge_base",
            "description": (
                "Check whether the knowledge base contains any indexed documents. "
                "Call this before searching if you are unsure whether documents exist. "
                "Returns the total document count and whether the KB is empty."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Returns the current UTC date and time. Use this when a customer asks "
                "about availability, business hours, or 24/7 support status. "
                "This assistant is always available — 24 hours a day, 7 days a week."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Use this when: (1) the knowledge base has no relevant information after "
                "at least one search attempt, (2) the customer explicitly asks for a "
                "human agent, or (3) the issue is complex/sensitive and needs human "
                "judgement. Generates a warm, empathetic handoff message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why escalation is needed (shown in the trace, not to the customer).",
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "A brief summary of the customer's issue to help the human "
                            "agent pick up context quickly."
                        ),
                    },
                },
                "required": ["reason", "summary"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt — upgraded for 24/7 robustness
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional, empathetic AI customer support agent available 24/7.

Your rules — follow all of them, every single response:
1. For any question about policies, orders, returns, shipping, warranties, or
   product topics: ALWAYS call search_docs first before answering.
2. For simple greetings ("hi", "hello", "hey", "good morning", etc.), respond
   warmly WITHOUT calling any tools — just introduce yourself and offer help.
3. If search results are weak (low scores or off-topic), call rephrase_and_retry
   with a better, more specific query.
4. Answer ONLY from tool results. Never invent facts, prices, or policies.
5. If you cannot find a confident answer after at least one search attempt,
   call escalate_to_human — do NOT guess.
6. You are available 24 hours a day, 7 days a week. If asked about availability,
   call get_current_time and confirm you are always here to help.
7. Keep answers concise, warm, and professional (1-3 paragraphs max).
8. Use conversation history to understand follow-up questions in context.\
"""

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(name: str, arguments: dict) -> str:
    """
    Dispatches a tool call and returns the result as a JSON string
    (the format Groq's function-calling protocol expects for tool messages).
    """
    logger.info("Executing tool '%s' with args: %s", name, arguments)

    if name == "search_docs":
        query = arguments.get("query", "")
        result = search(query, top_k=4)
        if result["error"]:
            return json.dumps({"error": result["error"], "matches": []})
        matches = result["matches"]
        formatted = [
            {
                "rank": i + 1,
                "source": m.get("source", "unknown"),
                "score": m.get("score"),
                "text": m.get("text", ""),
            }
            for i, m in enumerate(matches)
        ]
        return json.dumps({"matches": formatted, "count": len(formatted)})

    elif name == "rephrase_and_retry":
        new_query = arguments.get("new_query", "")
        result = search(new_query, top_k=4)
        if result["error"]:
            return json.dumps({"error": result["error"], "matches": []})
        matches = result["matches"]
        formatted = [
            {
                "rank": i + 1,
                "source": m.get("source", "unknown"),
                "score": m.get("score"),
                "text": m.get("text", ""),
            }
            for i, m in enumerate(matches)
        ]
        return json.dumps({"matches": formatted, "count": len(formatted)})

    elif name == "get_document_list":
        docs = list_documents()
        return json.dumps(
            {
                "documents": [
                    {"filename": d["filename"], "chunks": d["chunks"]}
                    for d in docs
                ],
                "total": len(docs),
            }
        )

    elif name == "check_knowledge_base":
        docs = list_documents()
        return json.dumps(
            {
                "document_count": len(docs),
                "is_empty": len(docs) == 0,
                "documents": [d["filename"] for d in docs],
            }
        )

    elif name == "get_current_time":
        now = datetime.now(timezone.utc)
        return json.dumps(
            {
                "utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "day_of_week": now.strftime("%A"),
                "availability": "24/7 — I am always available to help you.",
            }
        )

    elif name == "escalate_to_human":
        reason = arguments.get("reason", "")
        summary = arguments.get("summary", "")
        return json.dumps(
            {
                "escalated": True,
                "reason": reason,
                "summary": summary,
                "message": (
                    "I've noted your issue and a human support agent will be in touch "
                    "shortly. For immediate help, please contact our support team at "
                    "support@example.com or call 1-800-SUPPORT. We're sorry we couldn't "
                    "resolve this automatically and appreciate your patience."
                ),
            }
        )

    else:
        logger.warning("Unknown tool called: %s", name)
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Groq API call with tool support + retry logic
# ---------------------------------------------------------------------------


def _call_groq_with_tools(messages: list[dict]) -> dict:
    """
    Calls Groq with the full message history and tool definitions.

    Retries up to _GROQ_MAX_RETRIES times on HTTP 429 (rate limit) and
    HTTP 503 (service unavailable) with exponential back-off.

    Returns the raw response dict or {"error": str}.
    """
    if not GROQ_API_KEY:
        return {
            "error": (
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to your .env file."
            )
        }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False,
    }

    last_error: str = "Unknown error"

    for attempt in range(_GROQ_MAX_RETRIES + 1):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        except requests.exceptions.ConnectionError:
            return {"error": "Could not connect to Groq API. Check your internet connection."}
        except requests.exceptions.Timeout:
            last_error = "Groq API timed out."
            if attempt < _GROQ_MAX_RETRIES:
                delay = _GROQ_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Groq timeout on attempt %d — retrying in %.1fs", attempt + 1, delay)
                time.sleep(delay)
                continue
            return {"error": last_error}
        except requests.exceptions.RequestException as exc:
            return {"error": f"Unexpected error contacting Groq: {exc}"}

        if resp.status_code == 401:
            return {"error": "Invalid GROQ_API_KEY. Check your key at https://console.groq.com"}

        if resp.status_code in (429, 503):
            # Rate-limited or temporarily unavailable — back off and retry
            delay = _GROQ_RETRY_BASE_DELAY * (2 ** attempt)
            last_error = f"Groq returned HTTP {resp.status_code} (attempt {attempt + 1})."
            if attempt < _GROQ_MAX_RETRIES:
                logger.warning(
                    "Groq HTTP %d on attempt %d — retrying in %.1fs",
                    resp.status_code, attempt + 1, delay,
                )
                time.sleep(delay)
                continue
            return {
                "error": (
                    "Our AI service is currently under high load. "
                    "Please try again in a moment."
                )
            }

        if resp.status_code != 200:
            full_error = resp.text
            # Try to detect XML-style tool call failure and recover
            if "failed_generation" in full_error:
                try:
                    import re as _re
                    err_obj = json.loads(full_error)
                    failed_gen = err_obj["error"]["failed_generation"]
                    m = _re.search(r'<function=(\w+)(\{.*?\})', failed_gen, _re.DOTALL)
                    if m:
                        fn_name = m.group(1)
                        fn_args = m.group(2)
                        logger.warning("Recovering XML tool call: %s %s", fn_name, fn_args)
                        return {
                            "choices": [{
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [{
                                        "id": "recovered-001",
                                        "type": "function",
                                        "function": {"name": fn_name, "arguments": fn_args}
                                    }]
                                }
                            }]
                        }
                except Exception as parse_exc:
                    logger.debug("Recovery parse failed: %s", parse_exc)
            return {"error": f"Groq returned HTTP {resp.status_code}: {resp.text[:1000]}"}

        try:
            return resp.json()
        except Exception as exc:
            return {"error": f"Could not parse Groq response: {exc}"}

    return {"error": last_error}


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def run_agent(
    user_message: str,
    session_id: str,
) -> dict:
    """
    Runs the ReAct agent loop for one user turn.

    Returns:
        {
          "answer":      str | None,
          "error":       str | None,
          "escalated":   bool,
          "trace":       list[dict],   # tool calls + observations
          "timing":      {"total_ms": int, "iterations": int},
          "session_id":  str,
        }
    """
    t_start = time.perf_counter()
    session = get_or_create_session(session_id)
    trace: list[dict] = []
    escalated = False

    # Append the new user message to the session history
    session["history"].append({"role": "user", "content": user_message})

    # Build the full message list for this request:
    # system prompt + entire conversation history (gives the agent memory)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + session["history"]

    answer: str | None = None
    error: str | None = None
    iterations = 0

    # ── ReAct loop ────────────────────────────────────────────────────────
    for iteration in range(MAX_ITERATIONS):
        iterations += 1
        logger.info("Agent iteration %d / %d", iteration + 1, MAX_ITERATIONS)

        groq_resp = _call_groq_with_tools(messages)

        if "error" in groq_resp:
            error = groq_resp["error"]
            break

        choice = groq_resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")

        # ── Final answer — no more tool calls ────────────────────────────
        if finish_reason == "stop" or finish_reason == "length" or not message.get("tool_calls"):
            answer = (message.get("content") or "").strip() or "No answer received."
            session["history"].append({"role": "assistant", "content": answer})
            break

        # ── Tool call(s) ─────────────────────────────────────────────────
        tool_calls = message.get("tool_calls", [])

        # Add the assistant's tool-call message to the ongoing messages list
        # but NOT to session history (keeps history clean for next turn)
        messages.append(message)

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                tool_args = {}

            # Execute the tool
            tool_result = _execute_tool(tool_name, tool_args)

            # Record in trace for the frontend
            trace.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": json.loads(tool_result),
                    "iteration": iteration + 1,
                }
            )

            # Check for escalation signal
            try:
                result_obj = json.loads(tool_result)
                if result_obj.get("escalated"):
                    escalated = True
                    answer = result_obj.get("message", "Connecting you to a human agent.")
            except Exception:
                pass

            # Append tool result as a tool message
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                }
            )

        if escalated:
            # Persist escalation answer in history
            session["history"].append({"role": "assistant", "content": answer})
            break

    else:
        # Exhausted all iterations without a final answer — ask Groq to wrap up
        logger.warning("Agent hit iteration limit (%d), forcing final answer.", MAX_ITERATIONS)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Based on everything gathered so far, please give the customer "
                    "your best final answer. If you still can't help, escalate."
                ),
            }
        )
        groq_resp = _call_groq_with_tools(messages)
        if "error" not in groq_resp:
            msg = groq_resp.get("choices", [{}])[0].get("message", {})
            answer = msg.get("content", "").strip() or answer
        if answer:
            session["history"].append({"role": "assistant", "content": answer})

    total_ms = int((time.perf_counter() - t_start) * 1000)

    return {
        "answer": answer,
        "error": error,
        "escalated": escalated,
        "trace": trace,
        "timing": {"total_ms": total_ms, "iterations": iterations},
        "session_id": session_id,
    }
