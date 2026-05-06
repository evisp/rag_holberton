from app.services.rag_chain import ask


QUESTIONS = [
    "How can I transfer to another campus?",
    "What is the minimum average required for a campus transfer?",
    "How do I contact my campus director?",
    "What happens if I fail a trimester?",
    "How are projects evaluated at Holberton?",
]


def print_divider(char="─", width=55):
    print(char * width)


def print_result(result: dict, index: int):
    print(f"\n{'=' * 55}")
    print(f"  Question {index}: {result['question']}")
    print(f"{'=' * 55}")

    print(f"\n  Answer:")
    print(f"  {'─' * 50}")
    for line in result["answer"].splitlines():
        print(f"  {line}")

    print(f"\n  Sources ({len(result['sources'])} document(s)):")
    print(f"  {'─' * 50}")
    for source in result["sources"]:
        bar = _score_bar(source["score"])
        print(f"  {bar}  {source['filename']}")
        print(f"        {source['title']}")
        print(f"        relevance: {source['score']}")

    print(f"\n  Retrieved {result['top_k']} chunks from vector store")
    print(f"  Model: {result['model']}")


def _score_bar(score: float) -> str:
    """Visual relevance indicator."""
    if score >= 0.80:
        return "[████]"
    elif score >= 0.65:
        return "[███░]"
    elif score >= 0.50:
        return "[██░░]"
    else:
        return "[█░░░]"


def main():
    print("\n" + "★" * 55)
    print("  Holberton RAG — Phase 2 Test")
    print("  Retrieval + Generation")
    print("★" * 55)

    print("\nLoading retriever (FAISS index)...")

    for i, question in enumerate(QUESTIONS, 1):
        result = ask(question)
        print_result(result, i)

    print(f"\n{'=' * 55}")
    print("  All questions answered.")
    print(f"  RAG chain is working end to end.")
    print(f"  Ready for Phase 3 — Flask web interface.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()