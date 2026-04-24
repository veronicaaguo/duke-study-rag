"""
src/retrieval/reranker.py

Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Unlike bi-encoders (which embed query and doc separately),
a cross-encoder sees query+doc together and scores relevance directly.
Much more accurate but slower — used only on the top-20 candidates
from the first retrieval stage, not the whole corpus.
"""

from typing import List, Optional
from loguru import logger


class CrossEncoderReranker:
    """
    Wraps sentence-transformers CrossEncoder for reranking retrieval candidates.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info(f"CrossEncoder loaded: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed — reranker disabled")
            self.model = None

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """
        Score each candidate against the query, return top_k sorted by score.
        Falls back to original order if model unavailable.
        """
        if self.model is None or not candidates:
            return candidates[:top_k]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]
