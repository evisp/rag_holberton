import tiktoken
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Takes the output of load_documents() and splits each doc into chunks.
    Returns a flat list of chunk dicts ready for embedding.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    all_chunks = []

    for doc in documents:
        chunks = _chunk_text(doc["content"], encoder)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['filename']}::chunk_{i}",
                "filename": doc["filename"],
                "title": doc["title"],
                "filepath": doc["filepath"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content": chunk_text,
            })

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def _chunk_text(text: str, encoder, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping token-based chunks.
    """
    tokens = encoder.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start += chunk_size - overlap

    return chunks