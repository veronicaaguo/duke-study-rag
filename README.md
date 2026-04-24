# Duke Study RAG — A Trustworthy AI Study Assistant

## What it does

General-purpose LLMs have three problems when used for studying: they cap how many files you can upload, they lose context over long conversations, and they hallucinate — confidently stating things that aren't in your course materials. This project builds a RAG-based Q&A chatbot specifically designed for Duke course materials (lecture slides, PDFs, supplemental readings) that addresses all three.

Instead of pasting files into a prompt, the system indexes your entire course corpus into a vector database. At query time it retrieves only the most relevant chunks, grounds the LLM's response in those chunks, and shows you exactly which source it drew from — so you can verify every answer. The result is a study assistant you can actually trust when reviewing for exams.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=...

# 3. Ingest your course materials
python scripts/ingest.py --input data/raw/ --course "CS372"

# 4. Launch the app
streamlit run src/app/app.py
```

## Video links

- Demo video: [TBD]
- Technical walkthrough: [TBD]

## Evaluation

Full quantitative results are in `notebooks/evaluation.ipynb`. Summary:

| Pipeline | ROUGE-L | Faithfulness | Retrieval Recall@5 | Latency (s) |
|---|---|---|---|---|
| BM25 baseline | — | — | — | — |
| LangChain RAG (dense only) | — | — | — | — |
| Custom RAG (hybrid + rerank) | — | — | — | — |

*(To be filled in after experiments)*

## Project structure

```
duke-study-rag/
├── data/
│   ├── raw/          # Your course PDFs, slides, notes (not committed)
│   └── processed/    # Chunked + embedded documents
├── src/
│   ├── ingestion/    # Document loading, chunking strategies
│   ├── retrieval/    # Vector store, BM25, hybrid search, reranker
│   ├── generation/   # Prompt templates, LLM calls, CoT logic
│   ├── evaluation/   # Metrics: ROUGE, faithfulness, retrieval recall
│   └── app/          # Streamlit chat interface
├── notebooks/        # Experiments, ablation studies, error analysis
├── tests/            # Unit tests per module
├── scripts/          # ingest.py, evaluate.py, run_ablation.py
├── SETUP.md
├── ATTRIBUTION.md
└── requirements.txt
```
