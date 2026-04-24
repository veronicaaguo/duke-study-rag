# Duke Study RAG — A Trustworthy AI Study Assistant

## What it does

Using a general-purpose LLM (ChatGPT, Claude) as a study assistant has real limitations: file upload caps mean you can't load an entire semester's worth of slides at once, context windows degrade in quality as conversations grow long, and LLMs hallucinate — confidently stating things that aren't in your course materials. This project builds a RAG-based Q&A chatbot that addresses all three.

Instead of pasting files into a prompt, the system indexes your course corpus into a vector database. At query time it retrieves only the most relevant chunks, grounds the LLM's response in those chunks, and shows you exactly which source it drew from — so you can verify every answer. The result is a study assistant you can actually trust when reviewing for exams.

**Currently indexed: CS372 (Intro to Deep Learning, Spring 2026).** The app ships with 18 CS372 lectures already embedded in the vector database — launch it and it works immediately. To use it for a different course, replace the indexed corpus by running:

```bash
python scripts/ingest.py --input data/raw/<YOUR_COURSE>/ --course <COURSE_NAME>
```

This re-indexes your documents and rebuilds the BM25 index. The retrieval and generation logic is unchanged — only the source documents differ.

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

Full results and visualizations are in [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb). Ablation across 12 pipeline configurations (3 chunking strategies × 2 retrieval modes × 2 reranker settings) on the full 18-lecture CS372 corpus (382 chunks):

| Pipeline | Chunking | ROUGE-L | Faithfulness | Recall@5 | Latency (s) |
|---|---|---|---|---|---|
| Fixed chunks, dense only, no reranker *(baseline)* | fixed | 0.101 | 0.867 | 0.833 | 8.41 |
| Fixed chunks, dense only, + reranker | fixed | 0.103 | 0.867 | **0.933** | 7.66 |
| Fixed chunks, hybrid BM25, no reranker | fixed | 0.118 | 0.867 | 0.867 | 6.42 |
| Fixed chunks, hybrid BM25, + reranker | fixed | 0.105 | 0.800 | 0.900 | 6.08 |
| Sentence chunks, dense only, no reranker | sentence | 0.121 | 0.867 | 0.767 | 19.80 |
| Sentence chunks, hybrid BM25, + reranker | sentence | 0.105 | **0.933** | 0.833 | 7.33 |
| Semantic chunks, dense only, no reranker *(best ROUGE-L)* | semantic | **0.127** | 0.733 | 0.833 | 5.38 |
| Semantic chunks, hybrid BM25, no reranker | semantic | 0.117 | 0.667 | 0.700 | 5.61 |

**Key findings:**
- Semantic chunking achieves the best ROUGE-L (0.127) and fastest latency (5.38s) — grouping sentences by embedding similarity produces coherent, self-contained chunks that match query intent better
- Hybrid BM25 + dense retrieval boosts ROUGE-L +17% for fixed chunking (0.101 → 0.118) by combining keyword and semantic recall
- Cross-encoder reranking on sentence+hybrid achieves the highest faithfulness (0.933) — reranking surfaces chunks most relevant to the question, reducing hallucinated context
- Fixed-chunk + dense + reranker achieves the best Recall@5 (0.933) — larger chunks cast a wider net over relevant content
- End-to-end latency 5–20s; sentence chunking without reranking is the outlier at 19.8s due to the larger number of shorter chunks indexed

**Best pipeline results** (sentence chunking + hybrid BM25 + cross-encoder reranker, `text-embedding-3-small`, 31-question test set covering all 18 lectures):

| Metric | Score |
|---|---|
| ROUGE-L | **0.139** |
| Faithfulness (LLM-as-judge) | 0.839 |
| Retrieval Recall@5 | **0.871** |
| End-to-end latency | 8.5s |

*Note: Ablation study used `all-MiniLM-L6-v2` embeddings (384-dim) as a controlled baseline. Final pipeline upgraded to `text-embedding-3-small` (1536-dim), which improved ROUGE-L by +30% (0.107 → 0.139) and Recall@5 by +4.6% on the expanded 31-question set.*

*Full per-question error analysis and prompt style comparison in `notebooks/evaluation.ipynb`.*

## Limitations & Future Work

**In-browser file upload.** The most natural extension would be a file upload widget in the Streamlit sidebar — drag in your own PDFs and the system indexes them on the fly without touching the command line. The core challenge is ingestion latency: embedding a full semester's worth of slides (18 PDFs) takes 3–5 minutes, which is a poor user experience in a synchronous web UI. A production version would push ingestion to a background job (e.g., Celery + Redis) and show a live progress indicator, only making the index available once complete. This is left as future work.

**Multi-course index.** Currently the app serves a single course index. A natural extension is a course selector in the UI that switches between pre-built indexes without re-ingesting.

**Vision on more file types.** The current vision pipeline handles image-heavy PDF pages via GPT-4o. PPTX vision (diagrams embedded in slide objects) requires LibreOffice for rendering and is not enabled by default.

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
├── notebooks/        # Ablation results, error analysis, prompt comparison
├── scripts/          # ingest.py, evaluate.py, run_ablation.py
├── SETUP.md
├── ATTRIBUTION.md
└── requirements.txt
```
