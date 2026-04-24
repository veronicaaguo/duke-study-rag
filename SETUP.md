# Setup instructions

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key (get one at platform.openai.com)
- Your course materials as PDFs, PowerPoint files, or text documents

## Step 1 — Clone and install

```bash
git clone <your-repo-url>
cd duke-study-rag
pip install -r requirements.txt
```

## Step 2 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHROMA_PERSIST_DIR=data/processed/chroma
```

## Step 3 — Add your course materials

Drop your files into `data/raw/`. Supported formats: `.pdf`, `.pptx`, `.docx`, `.txt`, `.md`

Organize by course if you have multiple:
```
data/raw/
  CS372/
    lecture01_intro.pdf
    lecture02_linear_models.pdf
    ...
  STATS101/
    ...
```

## Step 4 — Ingest documents

```bash
python scripts/ingest.py --input data/raw/ --course CS372
```

This will chunk your documents, embed them, and store them in ChromaDB. Expect ~1–2 minutes per 100 pages.

## Step 5 — Run the app

```bash
streamlit run src/app/app.py
```

Open your browser to `http://localhost:8501`.

## Step 6 — (Optional) Run evaluation

```bash
python scripts/evaluate.py --course CS372 --test-set data/test_sets/CS372_qa.json
```

## Troubleshooting

- **"No module named chromadb"** — run `pip install -r requirements.txt` again inside your virtual environment
- **API key errors** — make sure `.env` exists and `OPENAI_API_KEY` is set correctly
- **Empty retrieval results** — make sure you ran `ingest.py` before starting the app
