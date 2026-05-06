from app.services.retriever import get_retriever
from app.services.generator import generate_answer
from app.config import TOP_K


def ask(question: str, top_k: int = TOP_K) -> dict:
    """
    Main entry point for the RAG system.
    
    Takes a plain English question, retrieves relevant chunks,
    generates a grounded answer, and returns everything structured.

    Returns:
        {
            "question":  the original question,
            "answer":    Gemini's grounded answer,
            "sources":   list of source documents with scores,
            "chunks":    raw retrieved chunks (for debugging),
            "top_k":     how many chunks were retrieved,
        }
    """
    if not question or not question.strip():
        return {
            "question": question,
            "answer": "Please provide a valid question.",
            "sources": [],
            "chunks": [],
            "top_k": top_k,
        }

    question = question.strip()

    retriever = get_retriever()
    chunks = retriever.search(question, top_k=top_k)

    result = generate_answer(question, chunks)

    return {
        "question": question,
        "answer":   result["answer"],
        "sources":  result["sources"],
        "chunks":   chunks,
        "top_k":    top_k,
        "model":    result["model"],
    }