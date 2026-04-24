"""
src/ingestion/chunker.py

Three chunking strategies for ablation study:
  1. fixed_size   — naive baseline, fixed token windows with overlap
  2. sentence     — splits on sentence boundaries, respects semantic units
  3. semantic     — groups sentences by embedding similarity (best quality, slowest)

The ablation in notebooks/ablation.ipynb compares all three on retrieval recall.
"""

from dataclasses import dataclass, field
from typing import List, Literal
import re
import os
from loguru import logger

from src.ingestion.loader import RawDocument


@dataclass
class Chunk:
    """A single retrievable chunk of text."""
    text: str
    source: str          # original file path
    course: str
    chunk_id: str        # "{source}::chunk_{n}"
    doc_type: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


ChunkStrategy = Literal["fixed", "sentence", "semantic"]


# ── 1. Fixed-size chunking (baseline) ────────────────────────────────────────

def chunk_fixed(doc: RawDocument, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    """
    Split text into fixed-size windows by character count with overlap.
    Simple and fast but may cut mid-sentence, harming coherence.
    Used as the ablation baseline.
    """
    text = doc.content
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=doc.source,
                course=doc.course,
                chunk_id=f"{doc.source}::chunk_{idx}",
                doc_type=doc.doc_type,
                chunk_index=idx,
                metadata={**doc.metadata, "strategy": "fixed"}
            ))
            idx += 1
        start += chunk_size - overlap
    return chunks


# ── 2. Sentence-aware chunking ────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex (no NLTK dependency needed)."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentence(doc: RawDocument, max_chars: int = 600, overlap_sentences: int = 1) -> List[Chunk]:
    """
    Group sentences into chunks that stay under max_chars.
    Never cuts mid-sentence — better coherence than fixed chunking.
    Overlap is measured in sentences, not characters.
    """
    sentences = _split_sentences(doc.content)
    chunks = []
    idx = 0
    i = 0
    while i < len(sentences):
        current_chunk = []
        current_len = 0
        while i < len(sentences) and current_len + len(sentences[i]) < max_chars:
            current_chunk.append(sentences[i])
            current_len += len(sentences[i])
            i += 1
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                source=doc.source,
                course=doc.course,
                chunk_id=f"{doc.source}::chunk_{idx}",
                doc_type=doc.doc_type,
                chunk_index=idx,
                metadata={**doc.metadata, "strategy": "sentence"}
            ))
            idx += 1
            # step back for overlap
            i = max(0, i - overlap_sentences)
    return chunks


# ── 3. Semantic chunking ──────────────────────────────────────────────────────

def chunk_semantic(doc: RawDocument, similarity_threshold: float = 0.85, max_chars: int = 800) -> List[Chunk]:
    """
    Group sentences together as long as their embeddings are similar.
    When similarity drops below threshold, start a new chunk.
    Produces the most coherent chunks but requires embedding calls.

    Note: imports sentence_transformers lazily to avoid cost when not needed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning("sentence-transformers not installed, falling back to sentence chunking")
        return chunk_sentence(doc, max_chars=max_chars)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = _split_sentences(doc.content)
    if not sentences:
        return []

    embeddings = model.encode(sentences, normalize_embeddings=True)

    chunks = []
    idx = 0
    current_group = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        if sim >= similarity_threshold and current_len + len(sentences[i]) < max_chars:
            current_group.append(sentences[i])
            current_len += len(sentences[i])
        else:
            chunks.append(Chunk(
                text=" ".join(current_group),
                source=doc.source,
                course=doc.course,
                chunk_id=f"{doc.source}::chunk_{idx}",
                doc_type=doc.doc_type,
                chunk_index=idx,
                metadata={**doc.metadata, "strategy": "semantic", "similarity_threshold": similarity_threshold}
            ))
            idx += 1
            current_group = [sentences[i]]
            current_len = len(sentences[i])

    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            source=doc.source,
            course=doc.course,
            chunk_id=f"{doc.source}::chunk_{idx}",
            doc_type=doc.doc_type,
            chunk_index=idx,
            metadata={**doc.metadata, "strategy": "semantic"}
        ))

    return chunks


# ── Dispatcher ────────────────────────────────────────────────────────────────

def chunk_document(doc: RawDocument, strategy: ChunkStrategy = "sentence", **kwargs) -> List[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc, **kwargs)
    elif strategy == "sentence":
        return chunk_sentence(doc, **kwargs)
    elif strategy == "semantic":
        return chunk_semantic(doc, **kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_documents(docs: List[RawDocument], strategy: ChunkStrategy = "sentence", **kwargs) -> List[Chunk]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, strategy=strategy, **kwargs)
        all_chunks.extend(chunks)
        logger.info(f"{doc.metadata.get('filename', doc.source)}: {len(chunks)} chunks ({strategy})")
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks
