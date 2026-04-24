"""
src/retrieval/bm25_retriever.py

BM25 keyword-based retrieval — the "lexical" half of hybrid search.
Also serves as a standalone baseline to compare against dense retrieval.

BM25 catches exact keyword matches that dense retrieval often misses
(e.g., specific technical terms, acronyms, proper nouns).
"""

from typing import List, Optional
import pickle
from pathlib import Path
from loguru import logger

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk


class BM25Retriever:
    """
    Builds a BM25 index over chunk texts.
    Tokenization is simple whitespace+lowercase — adequate for course material.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built over {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 20, course_filter: Optional[str] = None) -> List[dict]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built — call .index() first")

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Build results, applying course filter if specified
        results = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if course_filter and chunk.course != course_filter:
                continue
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "score": float(score),
                "metadata": {"course": chunk.course, "doc_type": chunk.doc_type, **chunk.metadata},
                "_chunk_idx": idx,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.bm25 = data["bm25"]
        logger.info(f"BM25 index loaded from {path} ({len(self.chunks)} chunks)")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()
