"""
scripts/ingest.py

Index course documents into ChromaDB + BM25.

Usage:
  python scripts/ingest.py --input data/raw/CS372 --course CS372
  python scripts/ingest.py --input data/raw/CS372 --course CS372 --no-vision
  python scripts/ingest.py --input data/raw/ --course CS372 --strategy semantic
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Index course documents")
    parser.add_argument("--input", required=True, help="Path to directory with course files")
    parser.add_argument("--course", required=True, help="Course name tag (e.g. CS372)")
    parser.add_argument("--strategy", default="sentence",
                        choices=["fixed", "sentence", "semantic"],
                        help="Chunking strategy (default: sentence)")
    parser.add_argument("--chunk-size", type=int, default=600,
                        help="Max chars per chunk (default: 600)")
    parser.add_argument("--no-vision", action="store_true",
                        help="Disable GPT-4o vision — text extraction only (faster, cheaper)")
    args = parser.parse_args()

    use_vision = not args.no_vision

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    if use_vision:
        logger.info("Vision mode ON — image-heavy pages will be described by GPT-4o")
        logger.info("Tip: results are cached in data/processed/vision_cache/ so you only pay once per page")
    else:
        logger.info("Vision mode OFF — text extraction only")

    # Load documents
    logger.info(f"Loading documents from {input_path}...")
    docs = load_directory(input_path, course=args.course, use_vision=use_vision)
    if not docs:
        logger.error("No documents found. Check file types and path.")
        sys.exit(1)

    # Chunk
    logger.info(f"Chunking with strategy='{args.strategy}'...")
    chunks = chunk_documents(docs, strategy=args.strategy, max_chars=args.chunk_size)

    # Index in ChromaDB (dense vectors)
    logger.info("Indexing in ChromaDB (dense)...")
    vs = VectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    vs.add_chunks(chunks)

    # Index in BM25 (sparse keyword)
    logger.info("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.index(chunks)
    bm25_path = f"data/processed/bm25_{args.course}.pkl"
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    bm25.save(bm25_path)

    logger.info(f"")
    logger.info(f"Done! Indexed {len(chunks)} chunks for course '{args.course}'.")
    logger.info(f"  ChromaDB total vectors : {vs.count()}")
    logger.info(f"  BM25 index saved to    : {bm25_path}")
    logger.info(f"  Vision cache           : data/processed/vision_cache/")


if __name__ == "__main__":
    main()
