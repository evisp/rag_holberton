import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.config import VECTOR_STORE_PATH


BM25_PATH = VECTOR_STORE_PATH / "bm25.pkl"


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.
    Splits on whitespace, strips punctuation, removes empty tokens.
    """
    tokens = text.lower().split()
    tokens = [t.strip(".,;:!?\"'()[]{}") for t in tokens]
    return [t for t in tokens if t]


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Builds a BM25 index from a list of chunk dicts.
    Each chunk must have a 'content' key.
    """
    corpus = [tokenize(chunk["content"]) for chunk in chunks]
    index  = BM25Okapi(corpus)
    return index


def save_bm25_index(index: BM25Okapi, chunks: list[dict]):
    """
    Saves the BM25 index and chunk order to disk.
    The chunk order matters — BM25 returns scores by position.
    """
    payload = {
        "index":  index,
        "chunks": [c["chunk_id"] for c in chunks],
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved BM25 index to {BM25_PATH}")


def load_bm25_index() -> tuple[BM25Okapi, list[str]]:
    """
    Loads BM25 index and chunk ID order from disk.
    Returns (index, chunk_ids).
    """
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. "
            "Run scripts/ingest.py first."
        )
    with open(BM25_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["index"], payload["chunks"]


def bm25_search(query: str, index: BM25Okapi,
                chunk_ids: list[str], metadata: list[dict],
                top_k: int = 20) -> list[dict]:
    """
    Searches the BM25 index for the query.
    Returns top_k chunks with a 'bm25_score' key added.
    """
    tokens = tokenize(query)
    scores = index.get_scores(tokens)

    id_to_meta = {m["chunk_id"]: m for m in metadata}

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []
    for idx, score in ranked:
        if score == 0:
            continue
        chunk_id = chunk_ids[idx]
        if chunk_id not in id_to_meta:
            continue
        chunk = id_to_meta[chunk_id].copy()
        chunk["bm25_score"] = float(score)
        results.append(chunk)

    return results