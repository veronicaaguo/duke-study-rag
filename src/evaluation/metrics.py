"""
src/evaluation/metrics.py

Evaluation metrics for the RAG pipeline.
Used in notebooks/evaluation.ipynb and scripts/evaluate.py.

Metrics (rubric item: "3+ distinct evaluation metrics"):
  1. rouge_l          — lexical overlap between generated and reference answer
  2. retrieval_recall — fraction of relevant chunks that appear in top-k results
  3. faithfulness     — whether the answer is grounded in retrieved context (via LLM judge)
  4. latency          — end-to-end query response time in seconds
"""

import time
import os
from typing import List, Optional
from loguru import logger


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Longest common subsequence F1 between prediction and reference."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, prediction)
    return result["rougeL"].fmeasure


# ── Retrieval Recall@K ────────────────────────────────────────────────────────

def compute_retrieval_recall(
    retrieved_chunks: List[dict],
    relevant_sources: List[str],
    k: int = 5
) -> float:
    """
    Fraction of known-relevant sources that appear in the top-k retrieved chunks.
    relevant_sources: list of source file paths known to contain the answer.
    """
    if not relevant_sources:
        return 0.0
    retrieved_sources = {c["source"] for c in retrieved_chunks[:k]}
    hits = sum(1 for s in relevant_sources if s in retrieved_sources)
    return hits / len(relevant_sources)


# ── Faithfulness (LLM-as-judge) ───────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to its source material.

Source material:
{context}

AI Answer:
{answer}

Is every claim in the AI answer directly supported by the source material?
Reply with only: "faithful" or "not faithful", followed by a one-sentence reason."""

def compute_faithfulness(answer: str, chunks: List[dict], model: str = "gpt-4o-mini") -> dict:
    """
    LLM-as-judge faithfulness check.
    Returns {'score': 0 or 1, 'label': str, 'reason': str}
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context = "\n\n".join(c["text"] for c in chunks[:5])
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip().lower()
    faithful = raw.startswith("faithful")
    return {
        "score": int(faithful),
        "label": "faithful" if faithful else "not faithful",
        "reason": raw,
    }


# ── Latency ───────────────────────────────────────────────────────────────────

class LatencyTimer:
    """Context manager for measuring end-to-end query latency."""
    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ── Full evaluation run ───────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline,           # HybridRetriever
    generator,          # StudyAssistant
    test_cases: List[dict],  # [{"question": str, "reference_answer": str, "relevant_sources": [...]}]
    top_k: int = 5,
) -> List[dict]:
    """
    Run full evaluation over a test set.
    Returns list of per-question result dicts.
    """
    results = []
    for i, case in enumerate(test_cases):
        logger.info(f"Evaluating case {i+1}/{len(test_cases)}: {case['question'][:60]}...")

        with LatencyTimer() as timer:
            chunks = pipeline.search(case["question"])
            result = generator.answer(case["question"], chunks)

        rouge = compute_rouge_l(result["answer"], case.get("reference_answer", ""))
        recall = compute_retrieval_recall(chunks, case.get("relevant_sources", []), k=top_k)
        faith = compute_faithfulness(result["answer"], chunks)

        results.append({
            "question": case["question"],
            "answer": result["answer"],
            "reference": case.get("reference_answer", ""),
            "rouge_l": round(rouge, 4),
            "retrieval_recall": round(recall, 4),
            "faithfulness": faith["score"],
            "faithfulness_label": faith["label"],
            "latency_s": round(timer.elapsed, 3),
            "chunks_used": len(chunks),
        })

    logger.info(f"Evaluation complete: {len(results)} cases")
    return results
