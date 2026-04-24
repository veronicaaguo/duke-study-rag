"""
scripts/run_ablation.py

Ablation study: systematically vary pipeline components and measure impact.

Design choices tested (rubric: "ablation study varying 2+ architectural decisions"):
  A. Chunking strategy:   fixed | sentence | semantic
  B. Retrieval mode:      dense_only | hybrid (dense + BM25)
  C. Reranker:            with | without

All combinations = 3 × 2 × 2 = 12 configurations.
Results are saved to data/ablation_results.csv for visualization in the notebook.

Usage:
  python scripts/run_ablation.py --course CS372 --test-set data/test_sets/CS372_qa.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from itertools import product

import pandas as pd
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


def build_pipeline(chunks, chunking_strategy, use_bm25, use_reranker, course):
    collection_name = f"ablation_{chunking_strategy}"
    vs = VectorStore(
        persist_dir=f"data/processed/ablation_chroma/{chunking_strategy}",
        collection_name=collection_name,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    vs.add_chunks(chunks)

    bm25 = BM25Retriever()
    bm25.index(chunks)

    reranker = CrossEncoderReranker() if use_reranker else None

    return HybridRetriever(
        vector_store=vs,
        bm25_retriever=bm25,
        reranker=reranker,
        top_k_retrieval=20,
        top_k_final=5,
        use_bm25=use_bm25,
        use_reranker=use_reranker,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", required=True)
    parser.add_argument("--test-set", required=True, help="Path to JSON test set")
    parser.add_argument("--input", default="data/raw/", help="Raw documents directory")
    parser.add_argument("--no-vision", action="store_true", help="Disable GPT-4o vision")
    args = parser.parse_args()

    # Load test set
    with open(args.test_set) as f:
        test_cases = json.load(f)
    logger.info(f"Loaded {len(test_cases)} test cases")

    # Load raw documents once
    docs = load_directory(Path(args.input), course=args.course, use_vision=not args.no_vision)

    # Ablation grid
    chunking_strategies = ["fixed", "sentence", "semantic"]
    bm25_options = [False, True]
    reranker_options = [False, True]

    all_results = []

    for strategy, use_bm25, use_reranker in product(chunking_strategies, bm25_options, reranker_options):
        config_name = f"{strategy} | {'hybrid' if use_bm25 else 'dense'} | {'rerank' if use_reranker else 'no-rerank'}"
        logger.info(f"\n{'='*60}\nConfig: {config_name}\n{'='*60}")

        # Chunk with this strategy
        chunks = chunk_documents(docs, strategy=strategy)

        # Build pipeline
        pipeline = build_pipeline(chunks, strategy, use_bm25, use_reranker, args.course)
        generator = StudyAssistant(prompt_style="cot")

        # Evaluate
        results = evaluate_pipeline(pipeline, generator, test_cases)

        # Aggregate
        avg = {
            "config": config_name,
            "chunking": strategy,
            "hybrid_search": use_bm25,
            "reranker": use_reranker,
            "rouge_l": round(sum(r["rouge_l"] for r in results) / len(results), 4),
            "retrieval_recall": round(sum(r["retrieval_recall"] for r in results) / len(results), 4),
            "faithfulness": round(sum(r["faithfulness"] for r in results) / len(results), 4),
            "latency_s": round(sum(r["latency_s"] for r in results) / len(results), 3),
            "n_cases": len(results),
        }
        all_results.append(avg)
        logger.info(f"Results: ROUGE-L={avg['rouge_l']} | Recall={avg['retrieval_recall']} | Faithful={avg['faithfulness']} | Latency={avg['latency_s']}s")

    # Save results
    out_path = "data/ablation_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(out_path, index=False)
    logger.info(f"\nAblation complete. Results saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
