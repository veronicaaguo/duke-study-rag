"""
src/generation/generator.py

Handles LLM calls with conversation history (multi-turn context management).

History is kept as a sliding window to avoid exceeding context limits.
The window size is configurable — this is one of the ablation variables.
"""

import os
from typing import List, Optional
from loguru import logger
from openai import OpenAI

from src.generation.prompts import get_prompt


class StudyAssistant:
    """
    Wraps OpenAI chat completion with:
    - Multi-turn conversation history (sliding window)
    - Configurable prompt style (direct / cot / socratic)
    - Source chunk tracking for UI display
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_style: str = "cot",
        max_history_turns: int = 6,   # each turn = 1 user + 1 assistant msg
    ):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.prompt_style = prompt_style
        self.max_history_turns = max_history_turns
        self.history: List[dict] = []

    def answer(self, question: str, chunks: List[dict]) -> dict:
        """
        Generate an answer grounded in retrieved chunks.
        Returns dict with 'answer', 'sources', 'model', 'prompt_style'.
        """
        # Trim history to sliding window
        trimmed_history = self._trim_history()

        # Build prompt
        system, messages = get_prompt(
            style=self.prompt_style,
            question=question,
            chunks=chunks,
            history=trimmed_history,
        )

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.2,   # low temperature for factual accuracy
            max_tokens=1024,
        )

        answer_text = response.choices[0].message.content

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer_text})

        # Extract unique sources for display
        sources = list({c["source"]: c for c in chunks}.values())

        logger.debug(f"Generated answer ({len(answer_text)} chars) from {len(chunks)} chunks")

        return {
            "answer": answer_text,
            "sources": sources,
            "model": self.model,
            "prompt_style": self.prompt_style,
            "chunks_used": len(chunks),
        }

    def reset_history(self) -> None:
        """Clear conversation history (new topic/session)."""
        self.history = []
        logger.info("Conversation history cleared")

    def _trim_history(self) -> List[dict]:
        """Keep only the last N turns to avoid context overflow."""
        max_messages = self.max_history_turns * 2  # user + assistant per turn
        return self.history[-max_messages:] if len(self.history) > max_messages else self.history
