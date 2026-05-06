import time
import google.generativeai as genai
from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL

genai.configure(api_key=GOOGLE_API_KEY)


def embed_documents(chunks: list[dict], batch_size: int = 100) -> list[dict]:
    """
    Takes chunks from chunker.py and adds an 'embedding' key to each.
    Processes in batches to respect API rate limits.
    Returns the same list with embeddings added in place.
    """
    total = len(chunks)
    print(f"Embedding {total} chunks in batches of {batch_size}...")

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        texts = [c["content"] for c in batch]

        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT",
            )

            for chunk, embedding in zip(batch, result["embedding"]):
                chunk["embedding"] = embedding

            print(f"  Embedded {min(batch_start + batch_size, total)}/{total}")

            # Respect free tier: 1500 RPM but be safe with batches
            if batch_start + batch_size < total:
                time.sleep(0.5)

        except Exception as e:
            print(f"Error embedding batch starting at {batch_start}: {e}")
            raise

    return chunks


def embed_query(text: str) -> list[float]:
    """
    Embeds a single user query for retrieval.
    Uses RETRIEVAL_QUERY task type — different from RETRIEVAL_DOCUMENT.
    """
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_QUERY",
        )
        return result["embedding"]

    except Exception as e:
        print(f"Error embedding query: {e}")
        raise