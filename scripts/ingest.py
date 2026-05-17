import json
import numpy as np
import faiss
from pathlib import Path
from app.config import VECTOR_STORE_PATH
from app.utils.loader import load_documents
from app.utils.chunker import chunk_documents
from app.services.embedder import embed_documents
from app.services.bm25_retriever import build_bm25_index, save_bm25_index


def build_faiss_index(chunks_with_embeddings: list[dict]) -> faiss.IndexFlatIP:
    """
    Builds a FAISS index from embedded chunks.
    Uses Inner Product (cosine similarity after normalization).
    """
    embeddings = np.array(
        [c["embedding"] for c in chunks_with_embeddings],
        dtype=np.float32
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"Built FAISS index: {index.ntotal} vectors, dimension {dimension}")
    return index


def save_faiss_index(index: faiss.IndexFlatIP, chunks: list[dict]):
    """
    Saves the FAISS index and metadata separately.
    metadata.json holds everything except the embedding vectors.
    """
    index_path    = VECTOR_STORE_PATH / "index.faiss"
    metadata_path = VECTOR_STORE_PATH / "metadata.json"

    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    metadata = []
    for c in chunks:
        metadata.append({
            "chunk_id":     c["chunk_id"],
            "filename":     c["filename"],
            "title":        c["title"],
            "filepath":     c["filepath"],
            "chunk_index":  c["chunk_index"],
            "total_chunks": c["total_chunks"],
            "content":      c["content"],
        })

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata for {len(metadata)} chunks to {metadata_path}")


def main():
    print("=" * 50)
    print("Holberton RAG — Ingestion Pipeline")
    print("=" * 50)

    print("\n[1/5] Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents found. Check your data/raw/ folder.")
        return

    print("\n[2/5] Chunking documents...")
    chunks = chunk_documents(documents)

    print("\n[3/5] Embedding chunks...")
    chunks_with_embeddings = embed_documents(chunks)

    print("\n[4/5] Building and saving FAISS index...")
    index = build_faiss_index(chunks_with_embeddings)
    save_faiss_index(index, chunks_with_embeddings)

    print("\n[5/5] Building and saving BM25 index...")
    bm25_index = build_bm25_index(chunks_with_embeddings)
    save_bm25_index(bm25_index, chunks_with_embeddings)

    print("\n" + "=" * 50)
    print("Ingestion complete!")
    print(f"  Documents : {len(documents)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Vectors   : {index.ntotal}")
    print(f"  BM25 docs : {len(chunks)}")
    print(f"  Saved to  : {VECTOR_STORE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()