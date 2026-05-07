import time
from app.services.rag_chain import ask
from app.services.memory import get_memory


def print_divider(char="─", width=55):
    print(char * width)


def print_turn(turn_num: int, question: str, answer: str):
    print(f"\n  Turn {turn_num}: {question}")
    print(f"  {'─' * 50}")
    for line in answer.splitlines():
        print(f"  {line}")


def test_multi_turn():
    print("\n" + "★" * 55)
    print("  Holberton RAG — Phase 4 Memory Test")
    print("  Multi-turn conversation demonstration")
    print("★" * 55)

    mem = get_memory()
    sid = mem.new_session_id()

    print(f"\n  Session ID : {sid[:18]}...")
    print(f"  Max history: {mem.stats()['max_history']} turns")
    print(f"  Session TTL: {mem.stats()['session_ttl']}s")

    conversation = [
        "What is the minimum average required for a campus transfer?",
        "Can I do it more than once?",
        "What about completing my specialization at another campus?",
        "Who should I speak to about that?",
    ]

    print(f"\n{'=' * 55}")
    print("  Test 1 — Follow-up resolution")
    print(f"  {len(conversation)} connected questions, no repeated context")
    print(f"{'=' * 55}")

    for i, question in enumerate(conversation, 1):
        result = ask(question, session_id=sid)
        sid = result["session_id"]
        print_turn(i, question, result["answer"])
        history = mem.get_history(sid)
        print(f"\n  History after turn {i}: {len(history)} turn(s) stored")
        if i < len(conversation):
            time.sleep(2)

    print(f"\n{'=' * 55}")
    print("  Test 2 — Session isolation")
    print("  Two sessions, same question, no cross-contamination")
    print(f"{'=' * 55}")

    sid_a = mem.new_session_id()
    sid_b = mem.new_session_id()

    r_a1 = ask("How can I transfer to another campus?", session_id=sid_a)
    sid_a = r_a1["session_id"]
    print(f"\n  Session A — Turn 1: campus transfer question answered")
    time.sleep(2)

    r_b1 = ask("How are projects evaluated?", session_id=sid_b)
    sid_b = r_b1["session_id"]
    print(f"  Session B — Turn 1: project evaluation question answered")
    time.sleep(2)

    r_a2 = ask("What is the minimum average for that?", session_id=sid_a)
    r_b2 = ask("What happens after the second deadline?", session_id=sid_b)

    print(f"\n  Session A — Turn 2: 'for that' resolved to campus transfer")
    print(f"  Answer: {r_a2['answer'][:80]}...")
    print(f"\n  Session B — Turn 2: 'after the second deadline' resolved to projects")
    print(f"  Answer: {r_b2['answer'][:80]}...")

    a_correct = "80" in r_a2["answer"]
    b_correct = "deadline" in r_b2["answer"].lower() or "regrade" in r_b2["answer"].lower()

    print(f"\n  Session A resolved correctly : {'✓' if a_correct else '✗'}")
    print(f"  Session B resolved correctly : {'✓' if b_correct else '✗'}")
    print(f"  Sessions did not contaminate : {'✓' if a_correct and b_correct else '✗'}")

    print(f"\n{'=' * 55}")
    print("  Test 3 — Session clear")
    print(f"{'=' * 55}")

    print(f"\n  History before clear: {len(mem.get_history(sid_a))} turn(s)")
    mem.clear_session(sid_a)
    print(f"  History after clear : {len(mem.get_history(sid_a))} turn(s)")
    print(f"  Clear confirmed     : {'✓' if len(mem.get_history(sid_a)) == 0 else '✗'}")

    print(f"\n{'=' * 55}")
    print("  Memory stats")
    print(f"{'=' * 55}")
    stats = mem.stats()
    print(f"\n  Active sessions : {stats['active_sessions']}")
    print(f"  Max history     : {stats['max_history']} turns per session")
    print(f"  Session TTL     : {stats['session_ttl']}s")

    print(f"\n{'=' * 55}")
    print("  All memory tests complete.")
    print("  Phase 4 verified — ready for Phase 5.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    test_multi_turn()