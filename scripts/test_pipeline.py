import json
import numpy as np
import faiss
from app.config import VECTOR_STORE_PATH
from app.utils.loader import load_documents
from app.utils.chunker import chunk_documents
from app.services.embedder import embed_query


def print_section(title: str):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def test_loader():
    print_section("STEP 1 — Document Loader")
    docs = load_documents()

    print(f"\n  Total documents loaded : {len(docs)}")
    print(f"\n  {'#':<4} {'Filename':<40} {'Words'}")
    print(f"  {'-'*4} {'-'*40} {'-'*5}")
    for i, doc in enumerate(docs[:10]):
        words = len(doc["content"].split())
        print(f"  {i+1:<4} {doc['filename']:<40} {words}")

    if len(docs) > 10:
        print(f"  ... and {len(docs) - 10} more documents")

    print(f"\n  Sample content from '{docs[0]['filename']}':")
    print(f"  {'─'*50}")
    preview = docs[0]["content"][:300].replace("\n", "\n  ")
    print(f"  {preview}...")

    return docs


def test_chunker(docs):
    print_section("STEP 2 — Chunker")
    chunks = chunk_documents(docs)

    print(f"\n  Total chunks created : {len(chunks)}")

    multi = [c for c in chunks if c["total_chunks"] > 1]
    print(f"  Docs split >1 chunk  : {len(multi)}")

    sizes = [len(c["content"].split()) for c in chunks]
    print(f"  Avg words per chunk  : {sum(sizes) // len(sizes)}")
    print(f"  Min words per chunk  : {min(sizes)}")
    print(f"  Max words per chunk  : {max(sizes)}")

    print(f"\n  Sample chunk:")
    print(f"  {'─'*50}")
    c = chunks[0]
    print(f"  chunk_id    : {c['chunk_id']}")
    print(f"  title       : {c['title']}")
    print(f"  chunk       : {c['chunk_index'] + 1} of {c['total_chunks']}")
    print(f"  content     : {c['content'][:200]}...")

    return chunks


def test_vector_store():
    print_section("STEP 3 — FAISS Vector Store")

    index_path = VECTOR_STORE_PATH / "index.faiss"
    metadata_path = VECTOR_STORE_PATH / "metadata.json"

    if not index_path.exists():
        print("\n  ERROR: index.faiss not found. Run scripts/ingest.py first.")
        return None, None

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"\n  Vectors in index     : {index.ntotal}")
    print(f"  Vector dimension     : {index.d}")
    print(f"  Metadata records     : {len(metadata)}")
    print(f"  Index file size      : {index_path.stat().st_size / 1024:.1f} KB")
    print(f"  Metadata file size   : {metadata_path.stat().st_size / 1024:.1f} KB")

    return index, metadata


def test_retrieval(index, metadata):
    print_section("STEP 4 — Retrieval Test (live query)")

    queries = [
        "How can I transfer to another campus?",
        "What is the grading policy?",
        "How do I contact my mentor?",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  {'─'*50}")

        query_embedding = np.array([embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, k=3)

        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            if idx == -1:
                continue
            chunk = metadata[idx]
            print(f"  [{rank}] score={score:.4f} | {chunk['filename']}")
            print(f"      {chunk['content'][:120].strip()}...")


def main():
    print("\n" + "★" * 55)
    print("  Holberton RAG — Pipeline Test & Demo")
    print("★" * 55)

    docs   = test_loader()
    chunks = test_chunker(docs)
    index, metadata = test_vector_store()

    if index is not None:
        test_retrieval(index, metadata)

    print_section("ALL TESTS COMPLETE")
    print(f"\n  Pipeline summary:")
    print(f"    Documents  →  {len(docs)}")
    print(f"    Chunks     →  {len(chunks)}")
    print(f"    Vectors    →  {index.ntotal if index else 'N/A'}")
    print(f"\n  RAG system is ready for Phase 2 (retrieval + Gemini)\n")


if __name__ == "__main__":
    main()