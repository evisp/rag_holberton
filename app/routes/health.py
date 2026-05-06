from flask import Blueprint, jsonify
from app.services.retriever import get_retriever
from app.services.memory import get_memory
from app.config import LLM_MODEL, EMBEDDING_MODEL, VECTOR_STORE_PATH

health_bp = Blueprint("health", __name__)


@health_bp.route("/health")
def health():
    try:
        retriever    = get_retriever()
        index_ok     = retriever.index is not None
        vector_count = retriever.index.ntotal if index_ok else 0
    except Exception:
        index_ok     = False
        vector_count = 0

    memory_stats = get_memory().stats()

    return jsonify({
        "status": "ok" if index_ok else "degraded",
        "vector_store": {
            "loaded":  index_ok,
            "vectors": vector_count,
            "path":    str(VECTOR_STORE_PATH),
        },
        "memory": memory_stats,
        "models": {
            "llm":       LLM_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
    })