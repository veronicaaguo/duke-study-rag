# Attribution

## AI Development Tools

This project was developed with assistance from Claude Code (Anthropic). GitHub Copilot was used for inline autocomplete on boilerplate.

### What was AI-generated

- Initial file scaffolding and directory structure
- Boilerplate `__init__.py` files and import blocks
- Initial docstrings and type hints (reviewed and corrected throughout)
- First-pass `loader.py` for PDF/PPTX text extraction (substantially extended)

### What was substantially reworked or debugged

- **`src/ingestion/chunker.py`** — AI suggested only fixed-size chunking. The sentence-aware and semantic chunking strategies were designed manually. Two infinite-loop bugs were found and fixed during development: (1) when a single sentence exceeded `max_chars`, `current_chunk` stayed empty and `i` never advanced; (2) the sentence overlap step `i -= overlap_sentences` reset `i` to `start_i` when a chunk contained exactly one sentence, looping forever. Both required careful trace-through of the loop logic to identify.
- **`src/retrieval/hybrid.py`** — AI generated a naive score-averaging approach. Replaced with Reciprocal Rank Fusion (RRF) based on Cormack et al. (2009), with k=60 smoothing constant. RRF is more robust because it is rank-based rather than score-based, avoiding scale mismatches between dense cosine similarity and BM25 scores.
- **`src/retrieval/vector_store.py`** — AI defaulted to OpenAI `text-embedding-3-small`. Switched to local `all-MiniLM-L6-v2` via `SentenceTransformerEmbeddingFunction`; added env-var fallback logic and `tqdm` progress bar. Also fixed a variable shadowing bug where `embedding_model` (parameter) was used instead of `self.embedding_model` in the branch logic. The local model was kept as default even after the original ingestion issue was resolved: `all-MiniLM-L6-v2` eliminates API round-trip latency during the embed-at-query-time path, costs nothing per call, and achieved strong retrieval performance on this corpus (Recall@5 up to 0.933 in ablation). `text-embedding-3-small` (1536-dim) would likely improve recall marginally on semantic edge cases, but the tradeoff — incurring API latency on every retrieval call and invalidating the completed ablation results — was not worthwhile for this project scope.
- **`src/evaluation/metrics.py`** — evaluation harness designed from scratch using RAGAS methodology. The faithfulness metric uses GPT-4o-mini as an LLM judge rather than the RAGAS framework directly, to avoid dependency on the RAGAS inference pipeline.
- **`scripts/run_ablation.py`** — AI generated a sequential loop; restructured to use `itertools.product` for the 3×2×2 grid and to load raw documents once outside the loop (avoiding re-loading 18 PDFs per config).

### What was written from scratch

- `src/retrieval/reranker.py` — cross-encoder reranking pipeline with graceful fallback
- All three prompt templates in `src/generation/prompts.py` (direct, chain-of-thought, Socratic)
- `data/test_sets/CS372_qa.json` — 15 Q&A pairs manually authored from CS372 lecture materials, each annotated with `relevant_sources` for recall evaluation
- Design of the multi-stage pipeline architecture (vision → chunk → embed → BM25+dense → RRF → rerank → generate)

### What I had to debug, fix, or rework beyond AI suggestions

- ChromaDB 1.x API changes (collection creation with embedding functions behaves differently from 0.x; had to trace collection persistence issues)
- Sentence chunker infinite loops (described above)
- `vector_store.py` variable shadowing causing wrong embedding model to be used
- App's BM25 loader hardcoded to `bm25_default.pkl` — fixed to glob for any `bm25_*.pkl` in `data/processed/`
- `run_ablation.py` had no `--no-vision` flag, causing unexpected GPT-4o API calls during ablation; added flag and plumbed `use_vision` parameter

## External Sources & References

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — Lewis et al., 2020 (RAG architecture)
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" — Es et al., 2023 (faithfulness metric design)
- "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" — Cormack et al., 2009 (RRF fusion with k=60)
- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" — Reimers & Gurevych, 2019 (all-MiniLM-L6-v2 architecture)
- LangChain documentation: https://python.langchain.com/
- ChromaDB documentation: https://docs.trychroma.com/
- Sentence Transformers documentation: https://www.sbert.net/
