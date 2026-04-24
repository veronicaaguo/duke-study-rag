# Attribution

## AI development tools

This project was developed with assistance from Claude (Anthropic) and GitHub Copilot.

### What was AI-generated

- Boilerplate code structure and file scaffolding (modified to fit project needs)
- Initial docstrings and type hints (reviewed and corrected throughout)
- Debugging assistance for ChromaDB configuration issues

### What was substantially reworked

- All chunking strategy logic in `src/ingestion/chunker.py` — AI suggestions used fixed-size chunking; the semantic and sentence-aware chunking strategies were designed and implemented manually
- The hybrid search fusion logic in `src/retrieval/hybrid.py` — AI generated a naive approach; RRF (Reciprocal Rank Fusion) weighting was implemented and tuned manually
- The evaluation harness in `src/evaluation/` — designed from scratch based on RAGAS documentation

### What was written from scratch

- `src/retrieval/reranker.py` — custom cross-encoder reranking pipeline
- All prompt templates in `src/generation/prompts.py`
- The ablation study runner in `scripts/run_ablation.py`
- Test cases in `data/test_sets/` — manually authored Q&A pairs from course materials

## External sources

- LangChain documentation: https://python.langchain.com/
- RAGAS documentation: https://docs.ragas.io/
- ChromaDB documentation: https://docs.trychroma.com/
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)
