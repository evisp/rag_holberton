from app.services.retriever import get_retriever
from app.services.generator import generate_answer
from app.services.memory import get_memory
from app.services.reranker import rerank
from app.config import TOP_K


def ask(question: str, session_id: str = None, top_k: int = TOP_K) -> dict:
    """
    Main entry point for the RAG system.

    Takes a plain English question plus an optional session_id,
    retrieves relevant chunks, reranks them, injects conversation
    history, generates a grounded answer, saves the turn, and
    returns everything structured.

    Pipeline:
      1. Hybrid search (FAISS + BM25 via RRF) → top-20 candidates
      2. Cross-encoder reranker → top-k (default 5)
      3. Inject conversation history into prompt
      4. Generate grounded answer with Gemini
      5. Save turn to session memory

    Returns:
        {
            "question":   the original question,
            "answer":     Gemini's grounded answer,
            "sources":    list of source documents with scores,
            "chunks":     reranked chunks used for generation,
            "top_k":      how many chunks were passed to Gemini,
            "model":      which model answered,
            "session_id": the session used,
        }
    """
    if not question or not question.strip():
        return {
            "question":   question,
            "answer":     "Please provide a valid question.",
            "sources":    [],
            "chunks":     [],
            "top_k":      top_k,
            "model":      None,
            "session_id": session_id,
        }

    question  = question.strip()
    memory    = get_memory()

    if not session_id:
        session_id = memory.new_session_id()

    history   = memory.get_history(session_id)
    retriever = get_retriever()

    candidates = retriever.search(question, top_k=20)
    chunks     = rerank(question, candidates, top_k=top_k)

    result = generate_answer(question, chunks, history=history)

    memory.add_turn(session_id, question, result["answer"])

    return {
        "question":   question,
        "answer":     result["answer"],
        "sources":    result["sources"],
        "chunks":     chunks,
        "top_k":      top_k,
        "model":      result["model"],
        "session_id": session_id,
    }