import json
import numpy as np
import faiss
from app.config import VECTOR_STORE_PATH, TOP_K
from app.services.embedder import embed_query
from app.services.bm25_retriever import load_bm25_index, bm25_search

FAISS_CANDIDATES = 20
BM25_CANDIDATES  = 20


def reciprocal_rank_fusion(
    faiss_results: list[dict],
    bm25_results:  list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum of 1 / (k + rank) across all lists.
    k=60 is the standard constant — dampens the impact of
    very high ranks without ignoring lower-ranked results.

    Returns a unified list sorted by fused score descending.
    """
    scores = {}
    chunks = {}

    for rank, chunk in enumerate(faiss_results, 1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunks[cid] = chunk

    for rank, chunk in enumerate(bm25_results, 1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        if cid not in chunks:
            chunks[cid] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, rrf_score in fused:
        chunk = chunks[cid].copy()
        chunk["score"] = round(rrf_score, 6)
        results.append(chunk)

    return results


class Retriever:
    def __init__(self):
        self.index      = None
        self.metadata   = []
        self.bm25_index = None
        self.bm25_ids   = []
        self._load()

    def _load(self):
        index_path    = VECTOR_STORE_PATH / "index.faiss"
        metadata_path = VECTOR_STORE_PATH / "metadata.json"
        bm25_path     = VECTOR_STORE_PATH / "bm25.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run scripts/ingest.py first."
            )

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if bm25_path.exists():
            self.bm25_index, self.bm25_ids = load_bm25_index()
            print(f"Retriever loaded: {self.index.ntotal} vectors, "
                  f"{len(self.metadata)} chunks, BM25 ready")
        else:
            print(f"Retriever loaded: {self.index.ntotal} vectors, "
                  f"{len(self.metadata)} chunks "
                  f"(BM25 not found — FAISS only)")

    def _faiss_search(self, question: str, top_k: int) -> list[dict]:
        """
        Searches the FAISS index using dense embeddings.
        """
        query_vector = np.array(
            [embed_query(question)],
            dtype=np.float32
        )
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx].copy()
            chunk["faiss_score"] = round(float(dist), 4)
            chunk["score"]       = chunk["faiss_score"]
            results.append(chunk)

        return results

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """
        Hybrid search: FAISS + BM25 fused with RRF.
        Falls back to FAISS-only if BM25 index is not available.
        Returns top_k results.
        """
        faiss_results = self._faiss_search(question, top_k=FAISS_CANDIDATES)

        if self.bm25_index is None:
            return faiss_results[:top_k]

        bm25_results = bm25_search(
            query      = question,
            index      = self.bm25_index,
            chunk_ids  = self.bm25_ids,
            metadata   = self.metadata,
            top_k      = BM25_CANDIDATES,
        )

        fused = reciprocal_rank_fusion(faiss_results, bm25_results)

        return fused[:top_k]


_retriever_instance = None


def get_retriever() -> Retriever:
    """
    Singleton retriever — loaded once, reused for every query.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


def reset_retriever():
    """
    Forces the singleton to reload on next call.
    Used by the auto-sync watcher after re-ingestion.
    """
    global _retriever_instance
    _retriever_instance = None
    print("Retriever reset — will reload on next query.")