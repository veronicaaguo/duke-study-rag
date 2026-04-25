"""
src/app/app.py

Streamlit chat interface for the Duke Study RAG assistant.

Features:
- Chat with course materials, source citations shown per answer
- Upload new documents directly from the UI (with vision toggle)
- Switch between courses and prompt styles
- Toggle hybrid search and reranker for live comparison
- Clear conversation history
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import StudyAssistant


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Duke Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Duke Study Assistant")
st.caption("Ask questions about your course materials. Every answer cites its source.")


# ── Cached components ─────────────────────────────────────────────────────────
# Defined before sidebar so the upload button handler can call them on rerun.

@st.cache_resource
def get_vector_store():
    return VectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma"),
    )


@st.cache_resource
def get_bm25():
    retriever = BM25Retriever()
    processed = Path("data/processed")
    pkl_files = sorted(processed.glob("bm25_*.pkl"), key=lambda p: p.stat().st_mtime) if processed.exists() else []
    if pkl_files:
        retriever.load(str(pkl_files[-1]))  # most recently modified
    return retriever


@st.cache_resource
def get_reranker():
    return CrossEncoderReranker(
        os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    course = st.text_input(
        "Course name", value="CS372",
        help="Filters retrieval to documents indexed under this course tag"
    )

    prompt_style = st.selectbox(
        "Answer style",
        options=["cot", "direct", "socratic"],
        format_func={
            "cot": "Step-by-step reasoning",
            "direct": "Direct answer",
            "socratic": "Socratic (asks follow-ups)"
        }.__getitem__,
    )

    st.divider()
    st.subheader("Retrieval pipeline")
    use_bm25 = st.toggle("Hybrid search (BM25 + dense)", value=True,
                         help="Combines keyword and semantic search via RRF fusion")
    use_reranker = st.toggle("Cross-encoder reranker", value=True,
                             help="Reranks top-20 candidates for higher precision")
    top_k = st.slider("Final sources to use", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Upload documents")

    use_vision_upload = st.toggle(
        "GPT-4o vision for image-heavy slides", value=True,
        help=(
            "When ON, slides/pages with little text are described by GPT-4o vision. "
            "Recommended for lecture slides with diagrams. "
            "Results are cached so each page is only billed once."
        )
    )

    uploaded_files = st.file_uploader(
        "Add course materials",
        type=["pdf", "pptx", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index uploaded files"):
        with st.spinner("Indexing... (vision API calls may take a moment)"):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    (Path(tmpdir) / f.name).write_bytes(f.read())
                docs = load_directory(
                    Path(tmpdir),
                    course=course,
                    use_vision=use_vision_upload,
                )
                chunks = chunk_documents(docs, strategy="sentence")
                vs = get_vector_store()
                vs.add_chunks(chunks)
                bm25 = get_bm25()
                bm25.index(chunks)
        total_vision = sum(d.metadata.get("vision_calls", 0) for d in docs)
        st.success(
            f"Indexed {len(uploaded_files)} file(s) → {len(chunks)} chunks"
            + (f" ({total_vision} vision API calls)" if total_vision else "")
        )
        st.rerun()

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        if "assistant" in st.session_state:
            st.session_state.assistant.reset_history()
        st.rerun()


def get_pipeline(use_bm25: bool, use_reranker: bool, top_k: int) -> HybridRetriever:
    return HybridRetriever(
        vector_store=get_vector_store(),
        bm25_retriever=get_bm25(),
        reranker=get_reranker() if use_reranker else None,
        top_k_retrieval=20,
        top_k_final=top_k,
        use_bm25=use_bm25,
        use_reranker=use_reranker,
    )


def render_answer(text: str) -> None:
    """
    Render an LLM answer in Streamlit with proper LaTeX support.

    GPT-4o-mini outputs LaTeX using \\(...\\) and \\[...\\] delimiters, but
    Streamlit's st.markdown() only renders $...$ and $$...$$ notation.
    This converts between the two so equations display correctly.
    """
    import re
    # Convert display math: \[...\]  →  $$...$$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Convert inline math: \(...\)  →  $...$
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    st.markdown(text)


def get_assistant(prompt_style: str) -> StudyAssistant:
    if (
        "assistant" not in st.session_state
        or st.session_state.get("_prompt_style") != prompt_style
    ):
        st.session_state.assistant = StudyAssistant(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prompt_style=prompt_style,
        )
        st.session_state["_prompt_style"] = prompt_style
    return st.session_state.assistant


# ── Chat UI ───────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            srcs = msg["sources"]
            with st.expander(f"Sources ({len(srcs)} chunk{'s' if len(srcs) != 1 else ''} matched)"):
                for i, chunk in enumerate(msg["sources"], 1):
                    fname = chunk["metadata"].get("filename", chunk["source"].split("/")[-1])
                    doc_type = chunk["metadata"].get("doc_type", "")
                    vision_badge = " 🔍" if "vision" in doc_type else ""
                    st.markdown(f"**[{i}] {fname}**{vision_badge}")
                    st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                    st.divider()

# Chat input
if question := st.chat_input("Ask a question about your course materials..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching course materials..."):
            pipeline = get_pipeline(use_bm25, use_reranker, top_k)
            assistant = get_assistant(prompt_style)
            chunks = pipeline.search(question, course_filter=course or None)

        if not chunks:
            answer = (
                "I couldn't find relevant information in your course materials. "
                "Try uploading documents first, or rephrase your question."
            )
            sources = []
        else:
            result = assistant.answer(question, chunks)
            answer = result["answer"]
            sources = chunks

        render_answer(answer)

        if sources:
            with st.expander(f"Sources ({len(sources)} chunk{'s' if len(sources) != 1 else ''} matched)"):
                for i, chunk in enumerate(sources, 1):
                    fname = chunk["metadata"].get("filename", chunk["source"].split("/")[-1])
                    doc_type = chunk["metadata"].get("doc_type", "")
                    vision_badge = " 🔍" if "vision" in doc_type else ""
                    st.markdown(f"**[{i}] {fname}**{vision_badge}")
                    st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
