import json
import numpy as np
import faiss
from app.config import VECTOR_STORE_PATH, TOP_K
from app.services.embedder import embed_query


class Retriever:
    def __init__(self):
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        index_path = VECTOR_STORE_PATH / "index.faiss"
        metadata_path = VECTOR_STORE_PATH / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run scripts/ingest.py first."
            )

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Retriever loaded: {self.index.ntotal} vectors, "
              f"{len(self.metadata)} chunks")

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the question and find the top-k most relevant chunks.
        Returns a list of chunk dicts with an added 'score' key.
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
            chunk["score"] = round(float(dist), 4)
            results.append(chunk)

        return results


_retriever_instance = None


def get_retriever() -> Retriever:
    """
    Returns a singleton retriever — index is loaded once,
    reused for every query. Important for Flask performance.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance