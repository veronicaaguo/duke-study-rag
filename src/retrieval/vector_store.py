"""
src/retrieval/vector_store.py

ChromaDB-backed vector store with OpenAI or sentence-transformer embeddings.
Supports adding chunks, dense similarity search, and persistence.
"""

from pathlib import Path
from typing import List, Optional
import os
from loguru import logger

import chromadb
from chromadb.utils import embedding_functions

from src.ingestion.chunker import Chunk


class VectorStore:
    """
    Wraps ChromaDB for persistent vector storage and dense retrieval.
    Embedding model is configurable for the ablation study comparing
    OpenAI text-embedding-3-small vs sentence-transformers bge-large.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "duke_study",
        embedding_model: str = "text-embedding-3-small",  # or "BAAI/bge-large-en-v1.5"
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Choose embedding function based on model name
        if embedding_model.startswith("text-embedding"):
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name=embedding_model,
            )
        else:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Vector store ready: {collection_name} ({self.collection.count()} docs)")

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> None:
        """Embed and store chunks in batches."""
        # Filter out already-indexed chunks
        existing_ids = set(self.collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already indexed, skipping.")
            return

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{
                    "source": c.source,
                    "course": c.course,
                    "doc_type": c.doc_type,
                    "chunk_index": c.chunk_index,
                    **c.metadata
                } for c in batch]
            )
            logger.info(f"Indexed batch {i // batch_size + 1}: {len(batch)} chunks")

        logger.info(f"Total indexed: {self.collection.count()}")

    def search(self, query: str, top_k: int = 20, course_filter: Optional[str] = None) -> List[dict]:
        """
        Dense vector similarity search.
        Returns list of dicts with 'text', 'source', 'score', 'metadata'.
        """
        where = {"course": course_filter} if course_filter else None
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source", ""),
                "score": 1 - dist,  # cosine distance → similarity
                "metadata": meta,
            })
        return hits

    def count(self) -> int:
        return self.collection.count()
