import json
import time
from pathlib import Path
import google.generativeai as genai
from app.config import GOOGLE_API_KEY, LLM_MODEL, LLM_FALLBACK_CHAIN
from app.services.rag_chain import ask

genai.configure(api_key=GOOGLE_API_KEY)

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH  = Path(__file__).parent / "results.json"

JUDGE_PROMPT = """You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.

You will be given:
- A question
- A ground truth answer (the correct answer from the source documents)
- A generated answer (what the RAG system produced)

Score the generated answer on two dimensions, each from 1 to 5:

FAITHFULNESS (1-5)
Does the generated answer stick to the facts?
5 = Completely accurate, nothing contradicts the ground truth
4 = Mostly accurate, minor omissions
3 = Partially accurate, some missing or imprecise information
2 = Several inaccuracies or unsupported claims
1 = Largely incorrect or contradicts the ground truth

RELEVANCE (1-5)
Does the generated answer actually address the question?
5 = Directly and completely answers the question
4 = Mostly answers the question, minor gaps
3 = Partially answers, misses some aspects
2 = Tangentially related but misses the point
1 = Does not answer the question at all

Respond ONLY with valid JSON in this exact format, nothing else:
{
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "reasoning": "<one sentence explaining your scores>"
}"""


def judge_answer(question: str, ground_truth: str, generated: str) -> dict:
    """
    Uses Gemini as a judge to score faithfulness and relevance.
    Tries the fallback chain if rate limited.
    """
    prompt = f"""Question: {question}

Ground truth: {ground_truth}

Generated answer: {generated}"""

    chain = [LLM_MODEL] + [m for m in LLM_FALLBACK_CHAIN if m != LLM_MODEL]

    for model_name in chain:
        for attempt in range(2):
            try:
                model    = genai.GenerativeModel(model_name=model_name)
                response = model.generate_content(
                    JUDGE_PROMPT + "\n\n" + prompt
                )
                text = response.text.strip()
                text = text.replace("```json", "").replace("```", "").strip()
                scores = json.loads(text)
                scores["model"] = model_name
                return scores

            except json.JSONDecodeError:
                return {
                    "faithfulness": 0,
                    "relevance":    0,
                    "reasoning":    "Judge returned invalid JSON",
                    "model":        model_name,
                }
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    print(f"      rate limited on {model_name}, waiting 30s...")
                    time.sleep(30)
                else:
                    print(f"      [{model_name}] error: {e}")
                    break

    return {
        "faithfulness": 0,
        "relevance":    0,
        "reasoning":    "All models failed",
        "model":        None,
    }


def run_evaluation():
    print("\n" + "★" * 55)
    print("  Holberton RAG — Evaluation")
    print("  Gemini-as-judge scoring")
    print("★" * 55)

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"\n  Loaded {len(eval_set)} evaluation questions")
    print(f"  Scoring: faithfulness + relevance (1–5 each)")
    print(f"  Pacing: 8s between questions, 60s pause every 5\n")

    results     = []
    total_faith = 0
    total_rel   = 0
    failed      = 0
    topic_scores = {}

    for i, item in enumerate(eval_set, 1):
        print(f"  [{i:02d}/{len(eval_set)}] {item['question'][:55]}...")

        rag_result = ask(item["question"])
        generated  = rag_result["answer"]

        if "could not find" in generated.lower() or "rate limited" in generated.lower():
            print(f"         ⚠ RAG returned no answer — skipping judge")
            scores = {
                "faithfulness": 0,
                "relevance":    0,
                "reasoning":    "RAG returned no answer",
                "model":        None,
            }
            failed += 1
        else:
            scores = judge_answer(
                item["question"],
                item["ground_truth"],
                generated,
            )

        result = {
            "id":           item["id"],
            "topic":        item["topic"],
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "generated":    generated,
            "sources":      [s["filename"] for s in rag_result.get("sources", [])],
            "faithfulness": scores["faithfulness"],
            "relevance":    scores["relevance"],
            "reasoning":    scores.get("reasoning", ""),
            "judge_model":  scores.get("model"),
        }
        results.append(result)

        f_score = scores["faithfulness"]
        r_score = scores["relevance"]
        total_faith += f_score
        total_rel   += r_score

        topic = item["topic"]
        if topic not in topic_scores:
            topic_scores[topic] = {"faithfulness": [], "relevance": []}
        topic_scores[topic]["faithfulness"].append(f_score)
        topic_scores[topic]["relevance"].append(r_score)

        bar_f = "█" * f_score + "░" * (5 - f_score)
        bar_r = "█" * r_score + "░" * (5 - r_score)
        print(f"         F [{bar_f}] {f_score}/5  "
              f"R [{bar_r}] {r_score}/5")
        print(f"         {scores.get('reasoning', '')[:70]}")

        if i < len(eval_set):
            if i % 5 == 0:
                print(f"\n  Pausing 60s every 5 questions to reset rate limits...\n")
                time.sleep(60)
            else:
                time.sleep(8)

    n = len(eval_set)
    avg_faith = round(total_faith / n, 2)
    avg_rel   = round(total_rel   / n, 2)

    summary = {
        "total_questions":  n,
        "failed_questions": failed,
        "avg_faithfulness": avg_faith,
        "avg_relevance":    avg_rel,
        "overall_score":    round((avg_faith + avg_rel) / 2, 2),
        "topic_breakdown":  {
            topic: {
                "avg_faithfulness": round(
                    sum(v["faithfulness"]) / len(v["faithfulness"]), 2),
                "avg_relevance": round(
                    sum(v["relevance"]) / len(v["relevance"]), 2),
            }
            for topic, v in topic_scores.items()
        },
        "results": results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 55}")
    print(f"  Evaluation complete")
    print(f"{'=' * 55}")
    print(f"\n  Questions evaluated : {n}")
    print(f"  Failed / skipped    : {failed}")
    print(f"\n  Avg faithfulness    : {avg_faith} / 5")
    print(f"  Avg relevance       : {avg_rel} / 5")
    print(f"  Overall score       : {summary['overall_score']} / 5")

    print(f"\n  Topic breakdown:")
    for topic, scores in summary["topic_breakdown"].items():
        print(f"    {topic:<22} "
              f"F={scores['avg_faithfulness']}  "
              f"R={scores['avg_relevance']}")

    print(f"\n  Results saved to: {RESULTS_PATH}")
    print(f"{'=' * 55}\n")

    return summary


if __name__ == "__main__":
    run_evaluation()