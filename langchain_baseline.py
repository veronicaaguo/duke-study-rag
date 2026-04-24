"""
src/retrieval/langchain_baseline.py

LangChain RetrievalQA baseline — the "Used RAG via framework" rubric item (5 pts).

This is the simple out-of-the-box RAG that the custom pipeline (hybrid.py) is
compared against in the ablation study. Deliberately minimal to show what the
custom pipeline improves upon.
"""

import os
from typing import List
from loguru import logger

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document


class LangChainRAGBaseline:
    """
    Minimal LangChain RAG: OpenAI embeddings → ChromaDB → GPT-4o-mini.
    No hybrid search, no reranking, no custom chunking.
    Used as the comparison baseline in evaluation.
    """

    def __init__(self, persist_dir: str, top_k: int = 5):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="langchain_baseline"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
        logger.info("LangChain RAG baseline initialized")

    def answer(self, question: str) -> dict:
        result = self.chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [{"text": d.page_content, "source": d.metadata.get("source", "")}
                        for d in result.get("source_documents", [])],
        }
