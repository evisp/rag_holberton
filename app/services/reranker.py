from sentence_transformers import CrossEncoder
from app.config import TOP_K

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_instance = None


def get_reranker() -> CrossEncoder:
    """
    Singleton cross-encoder — loaded once, reused for every query.
    Model is ~90MB, loading it per request would be very slow.
    """
    global _reranker_instance
    if _reranker_instance is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
        print("Reranker ready.")
    return _reranker_instance


def rerank(question: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Reranks a list of retrieved chunks using the cross-encoder.

    The cross-encoder reads the query and each chunk together as one
    input — unlike embedding similarity which encodes them separately.
    This joint reading produces much more accurate relevance scores.

    Steps:
      1. Build (query, chunk_content) pairs
      2. Score all pairs in one batch
      3. Sort by score descending
      4. Return top_k with 'rerank_score' added

    Args:
        question: the user's original question
        chunks:   candidate chunks from hybrid search (typically top-20)
        top_k:    how many to return after reranking (typically 5)

    Returns:
        top_k chunks sorted by rerank score, highest first
    """
    if not chunks:
        return chunks

    if len(chunks) <= top_k:
        return chunks

    reranker = get_reranker()

    pairs = [(question, chunk["content"]) for chunk in chunks]

    scores = reranker.predict(pairs)

    scored_chunks = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    top = scored_chunks[:top_k]

    for chunk in top:
        chunk["score"] = round(chunk["rerank_score"], 4)

    return top