"""
src/generation/prompts.py

Prompt templates for the study assistant.
Three prompt designs are evaluated in the ablation:
  1. direct       — straightforward answer with citations
  2. cot          — chain-of-thought: model reasons before answering
  3. socratic     — model asks a follow-up question to deepen understanding

The comparison of these three is the "prompt engineering" rubric item (3 pts).
"""

from typing import List


def format_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into numbered context block."""
    sections = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk["metadata"].get("filename", chunk["source"].split("/")[-1])
        sections.append(f"[Source {i}: {filename}]\n{chunk['text']}")
    return "\n\n---\n\n".join(sections)


# ── Prompt 1: Direct answer with citations ────────────────────────────────────

SYSTEM_DIRECT = """You are a study assistant for Duke University courses. 
You answer questions strictly based on the provided course materials.

Rules:
- Only use information from the provided sources
- Cite sources by their number (e.g. "According to [Source 2]...")
- If the answer is not in the sources, say so clearly — do not guess
- Be concise and precise, as if explaining to a fellow student"""

def prompt_direct(question: str, chunks: List[dict], history: List[dict]) -> List[dict]:
    context = format_context(chunks)
    messages = history.copy()
    messages.append({
        "role": "user",
        "content": f"Course materials:\n\n{context}\n\n---\n\nQuestion: {question}"
    })
    return messages


# ── Prompt 2: Chain-of-thought reasoning ─────────────────────────────────────

SYSTEM_COT = """You are a study assistant for Duke University courses.
You answer questions strictly based on the provided course materials.

Before answering, reason step by step:
1. Identify which sources are relevant to the question
2. Extract the key facts from those sources
3. Synthesize a clear, accurate answer
4. Cite your sources

If the answer is not in the sources, say so — do not guess."""

def prompt_cot(question: str, chunks: List[dict], history: List[dict]) -> List[dict]:
    context = format_context(chunks)
    messages = history.copy()
    messages.append({
        "role": "user",
        "content": (
            f"Course materials:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\n"
            f"Think step by step before answering."
        )
    })
    return messages


# ── Prompt 3: Socratic mode ───────────────────────────────────────────────────

SYSTEM_SOCRATIC = """You are a Socratic study assistant for Duke University courses.
Your goal is not just to answer questions but to deepen understanding.

When answering:
1. Provide the direct answer from the course materials (with citations)
2. Then ask one thoughtful follow-up question that would help the student 
   think more deeply about the concept — something that connects it to 
   adjacent ideas or tests understanding

Only use information from the provided sources. Do not guess."""

def prompt_socratic(question: str, chunks: List[dict], history: List[dict]) -> List[dict]:
    context = format_context(chunks)
    messages = history.copy()
    messages.append({
        "role": "user",
        "content": f"Course materials:\n\n{context}\n\n---\n\nQuestion: {question}"
    })
    return messages


# ── Prompt dispatcher ─────────────────────────────────────────────────────────

PROMPTS = {
    "direct": (SYSTEM_DIRECT, prompt_direct),
    "cot": (SYSTEM_COT, prompt_cot),
    "socratic": (SYSTEM_SOCRATIC, prompt_socratic),
}

def get_prompt(
    style: str,
    question: str,
    chunks: List[dict],
    history: List[dict]
) -> tuple[str, List[dict]]:
    """Returns (system_prompt, messages) for the given style."""
    if style not in PROMPTS:
        raise ValueError(f"Unknown prompt style: {style}. Choose from {list(PROMPTS.keys())}")
    system, builder = PROMPTS[style]
    messages = builder(question, chunks, history)
    return system, messages
