"""
src/retrieval/hybrid.py

Hybrid search: dense (vector) + sparse (BM25) retrieval fused with
Reciprocal Rank Fusion (RRF), then reranked by a cross-encoder.

Pipeline:
  query
    ├─► BM25 → top-20 lexical hits
    └─► ChromaDB dense → top-20 semantic hits
              └─► RRF fusion → merged top-20
                      └─► cross-encoder rerank → final top-5

This is the "custom RAG pipeline" rubric item (10 pts).
Each component is independently ablatable — see scripts/run_ablation.py.
"""

from typing import List, Optional
from loguru import logger

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker


def reciprocal_rank_fusion(
    result_lists: List[List[dict]],
    k: int = 60
) -> List[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) across all lists.
    k=60 is the standard value from the original RRF paper (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            key = hit["text"]  # use text as dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key] = hit

    merged = sorted(docs.values(), key=lambda h: scores[h["text"]], reverse=True)
    for hit in merged:
        hit["rrf_score"] = scores[hit["text"]]
    return merged


class HybridRetriever:
    """
    Orchestrates dense + BM25 + RRF + reranking into a single .search() call.
    Each stage is optional — set use_bm25=False or use_reranker=False to ablate.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k_retrieval: int = 20,
        top_k_final: int = 5,
        use_bm25: bool = True,
        use_reranker: bool = True,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.use_bm25 = use_bm25
        self.use_reranker = use_reranker

    def search(self, query: str, course_filter: Optional[str] = None) -> List[dict]:
        """
        Full hybrid search pipeline.
        Returns top_k_final chunks with text, source, and score.
        """
        # Stage 1: Dense retrieval
        dense_hits = self.vector_store.search(
            query, top_k=self.top_k_retrieval, course_filter=course_filter
        )
        logger.debug(f"Dense retrieval: {len(dense_hits)} hits")

        # Stage 2: BM25 retrieval (optional)
        if self.use_bm25:
            bm25_hits = self.bm25.search(
                query, top_k=self.top_k_retrieval, course_filter=course_filter
            )
            logger.debug(f"BM25 retrieval: {len(bm25_hits)} hits")
            merged = reciprocal_rank_fusion([dense_hits, bm25_hits])
        else:
            merged = dense_hits

        # Stage 3: Reranking (optional)
        candidates = merged[:self.top_k_retrieval]
        if self.use_reranker and self.reranker is not None:
            final = self.reranker.rerank(query, candidates, top_k=self.top_k_final)
            logger.debug(f"After reranking: {len(final)} chunks")
        else:
            final = candidates[:self.top_k_final]

        return final
