import time
import google.generativeai as genai
from app.config import GOOGLE_API_KEY, LLM_MODEL, LLM_FALLBACK_CHAIN

genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant for Holberton School students and staff.
You answer questions strictly based on the provided context excerpts from
Holberton's internal documents.

Rules you must follow:
- Answer only from the context provided. Do not use outside knowledge.
- If the context does not contain enough information to answer, say clearly:
  "I could not find a clear answer in the Holberton documents."
- Be concise and direct. Students want quick, clear answers.
- Never make up policies, deadlines, or procedures.
- If multiple documents are relevant, synthesize them into one clear answer.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        block = (
            f"[Source {i}: {chunk['filename']}]\n"
            f"{chunk['content'].strip()}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    return f"""Use the following excerpts from Holberton internal documents
to answer the question.

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Question: {question}

Answer:"""


def _try_model(model_name: str, prompt: str, retries: int = 2) -> str | None:
    """
    Try a single model with exponential backoff.
    Returns the answer string, or None if rate limited after retries.
    Raises on non-quota errors.
    """
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )

    for attempt in range(retries + 1):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            error_str = str(e)

            if "429" in error_str:
                if attempt < retries:
                    wait = 10 * (2 ** attempt)  # 10s, 20s
                    print(f"    [{model_name}] rate limited, "
                          f"retrying in {wait}s... ({attempt + 1}/{retries})")
                    time.sleep(wait)
                else:
                    print(f"    [{model_name}] quota exhausted, trying next model.")
                    return None
            else:
                raise

    return None


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Tries each model in the fallback chain until one succeeds.
    Returns structured response with answer, sources, and model used.
    """
    if not chunks:
        return {
            "answer": "I could not find any relevant documents for your question.",
            "sources": [],
            "model": None,
        }

    prompt = build_prompt(question, chunks)

    # Build chain: start from configured primary, then rest of fallbacks
    chain = [LLM_MODEL] + [m for m in LLM_FALLBACK_CHAIN if m != LLM_MODEL]

    answer = None
    model_used = None

    for model_name in chain:
        print(f"    → trying {model_name}...")
        try:
            result = _try_model(model_name, prompt)
            if result is not None:
                answer = result
                model_used = model_name
                break
        except Exception as e:
            print(f"    [{model_name}] error: {e}")
            continue

    if answer is None:
        answer = (
            "All available models are currently rate limited. "
            "Please wait a minute and try again."
        )

    return {
        "answer": answer,
        "sources": deduplicate_sources(chunks),
        "model": model_used,
    }


def deduplicate_sources(chunks: list[dict]) -> list[dict]:
    seen = {}
    for chunk in chunks:
        fname = chunk["filename"]
        if fname not in seen or chunk["score"] > seen[fname]["score"]:
            seen[fname] = {
                "filename": fname,
                "title": chunk["title"],
                "score": chunk["score"],
            }
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)