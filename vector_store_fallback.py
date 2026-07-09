"""
vector_store_fallback.py

In-memory vector store fallback when Endee isn't available.
Uses simple numpy cosine similarity — good enough for demo/development.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# In-memory storage
_vectors: dict[str, dict] = {}  # {id: {vector, meta}}
_index_exists = False


def ensure_index_exists() -> bool:
    global _index_exists
    _index_exists = True
    logger.info("Using in-memory vector store (Endee fallback)")
    return True


def insert_vectors(vectors_data: list[dict]) -> tuple[bool, str | None]:
    """
    Insert vectors into in-memory store.
    vectors_data: [{"id": str, "vector": list[float], "meta": str}, ...]
    """
    for item in vectors_data:
        _vectors[item["id"]] = {
            "vector": np.array(item["vector"], dtype=np.float32),
            "meta": item["meta"],
        }
    return True, None


def search_vectors(query_vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Cosine similarity search.
    Returns: [{"id": str, "meta": str, "similarity": float}, ...]
    """
    if not _vectors:
        return []

    query_np = np.array(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query_np)
    
    if query_norm == 0:
        return []

    results = []
    for vec_id, data in _vectors.items():
        stored_vec = data["vector"]
        stored_norm = np.linalg.norm(stored_vec)
        
        if stored_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_np, stored_vec) / (query_norm * stored_norm))
        
        results.append({
            "id": vec_id,
            "meta": data["meta"],
            "similarity": similarity,
        })
    
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def check_connection() -> bool:
    """Fallback always returns True."""
    return True
