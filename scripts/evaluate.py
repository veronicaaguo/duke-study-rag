"""
scripts/evaluate.py

Run the best pipeline configuration on the full test set and save
per-question results for error analysis in notebooks/evaluation.ipynb.

Usage:
  python scripts/evaluate.py --course CS372 --test-set data/test_sets/CS372_qa.json --input data/raw/CS372 --no-vision
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import StudyAssistant
from src.evaluation.metrics import evaluate_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--input", default="data/raw/")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--chunking", default="sentence",
                        choices=["fixed", "sentence", "semantic"])
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--out", default="data/eval_results_best_pipeline.json")
    args = parser.parse_args()

    with open(args.test_set) as f:
        test_cases = json.load(f)
    logger.info(f"Loaded {len(test_cases)} test cases")

    docs = load_directory(Path(args.input), course=args.course,
                          use_vision=not args.no_vision)
    chunks = chunk_documents(docs, strategy=args.chunking)

    vs = VectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma"),
        collection_name=f"eval_{args.chunking}",
    )
    vs.add_chunks(chunks)

    bm25 = BM25Retriever()
    bm25.index(chunks)

    use_bm25 = not args.no_bm25
    use_reranker = not args.no_reranker
    reranker = CrossEncoderReranker() if use_reranker else None

    pipeline = HybridRetriever(
        vector_store=vs,
        bm25_retriever=bm25,
        reranker=reranker,
        top_k_retrieval=20,
        top_k_final=5,
        use_bm25=use_bm25,
        use_reranker=use_reranker,
    )
    generator = StudyAssistant(prompt_style="cot")
    results = evaluate_pipeline(pipeline, generator, test_cases)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} results → {args.out}")

    avg_rouge  = sum(r["rouge_l"] for r in results) / len(results)
    avg_faith  = sum(r["faithfulness"] for r in results) / len(results)
    avg_recall = sum(r["retrieval_recall"] for r in results) / len(results)
    avg_lat    = sum(r["latency_s"] for r in results) / len(results)
    logger.info(f"ROUGE-L={avg_rouge:.4f} | Faithfulness={avg_faith:.2f} | "
                f"Recall@5={avg_recall:.4f} | Latency={avg_lat:.2f}s")


if __name__ == "__main__":
    main()
