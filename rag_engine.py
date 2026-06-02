"""
RAGEngine: Retrieval-Augmented Generation pipeline.
Indexes document chunks with FAISS and answers natural language questions
by retrieving relevant context and calling Claude via the Anthropic API.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class RetrievedChunk:
    """A single retrieved text chunk with metadata."""
    text: str
    paper_id: str
    paper_title: str
    chunk_index: int
    score: float


@dataclass
class RAGAnswer:
    """Structured answer from the RAG pipeline."""
    question: str
    answer: str
    sources: list[RetrievedChunk]
    model: str = ""

    def formatted(self) -> str:
        lines = [f"**Q:** {self.question}", "", f"**A:** {self.answer}", "", "**Sources:**"]
        seen: set[str] = set()
        for src in self.sources:
            if src.paper_title not in seen:
                seen.add(src.paper_title)
                lines.append(f"  • {src.paper_title} (score: {src.score:.3f})")
        return "\n".join(lines)


class RAGEngine:
    """
    RAG pipeline:
    1. Index chunks → FAISS (or brute-force cosine fallback).
    2. On query: embed question → retrieve top-k chunks.
    3. Build context prompt → call LLM → return answer.

    LLM backend: Anthropic Claude (claude-sonnet-4-20250514).
    API key is read from the ANTHROPIC_API_KEY environment variable.
    """

    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        embedding_generator,  # EmbeddingGenerator instance
        top_k: int = 5,
        api_key: Optional[str] = None,
    ):
        self.embedding_gen = embedding_generator
        self.top_k = top_k
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

        self._chunks: list[RetrievedChunk] = []
        self._embeddings: np.ndarray = np.empty((0,), dtype=np.float32)
        self._index: Optional[object] = None  # FAISS index

    # ------------------------------------------------------------------ #
    #  Index management                                                    #
    # ------------------------------------------------------------------ #

    def add_chunks(
        self,
        chunks: list[str],
        paper_id: str,
        paper_title: str,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add chunks from a paper to the index.

        Args:
            chunks: List of text chunks.
            paper_id: Identifier for the paper.
            paper_title: Human-readable title.
            embeddings: Pre-computed embeddings (shape: N×D). If None, they are computed.
        """
        if not chunks:
            return

        if embeddings is None:
            embeddings = self.embedding_gen.embed_batch(chunks)

        dim = embeddings.shape[1]

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            self._chunks.append(RetrievedChunk(
                text=chunk,
                paper_id=paper_id,
                paper_title=paper_title,
                chunk_index=i,
                score=0.0,
            ))

        if self._embeddings.size == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        self._index = None  # Invalidate

    def remove_paper(self, paper_id: str) -> None:
        """Remove all chunks belonging to a paper and rebuild index."""
        keep = [i for i, c in enumerate(self._chunks) if c.paper_id != paper_id]
        self._chunks = [self._chunks[i] for i in keep]
        self._embeddings = self._embeddings[keep] if self._embeddings.size else self._embeddings
        self._index = None

    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ #
    #  Question answering                                                  #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGAnswer:
        """
        Answer a natural language question using retrieved context.

        Args:
            question: The user's question.
            top_k: Override default retrieval count.

        Returns:
            RAGAnswer with answer text and source chunks.
        """
        k = top_k or self.top_k
        retrieved = self.retrieve(question, k)

        if not retrieved:
            return RAGAnswer(
                question=question,
                answer="No relevant documents found in the knowledge base.",
                sources=[],
            )

        context = self._build_context(retrieved)
        answer_text = self._call_llm(question, context)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=retrieved,
            model=self.MODEL,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query."""
        k = top_k or self.top_k
        if not self._chunks:
            return []

        query_vec = self.embedding_gen.embed_text(query)

        if FAISS_AVAILABLE:
            scores, indices = self._get_faiss_index().search(query_vec[np.newaxis, :], min(k, len(self._chunks)))
            scores, indices = scores[0], indices[0]
        else:
            all_scores = (self._embeddings @ query_vec).clip(-1, 1)
            indices = np.argsort(-all_scores)[:k]
            scores = all_scores[indices]

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            results.append(RetrievedChunk(
                text=chunk.text,
                paper_id=chunk.paper_id,
                paper_title=chunk.paper_title,
                chunk_index=chunk.chunk_index,
                score=float(score),
            ))
        return results

    # ------------------------------------------------------------------ #
    #  LLM                                                                 #
    # ------------------------------------------------------------------ #

    def _call_llm(self, question: str, context: str) -> str:
        if not self.api_key:
            return (
                "[LLM not configured — set ANTHROPIC_API_KEY] "
                f"Relevant context was found but cannot be summarised without an API key.\n\n"
                f"Context preview:\n{context[:500]}…"
            )

        system_prompt = (
            "You are a scientific research assistant. "
            "Answer the user's question using ONLY the provided research paper excerpts. "
            "If the answer cannot be determined from the context, say so clearly. "
            "Cite the paper titles when referencing specific information. "
            "Be concise and precise."
        )

        user_prompt = (
            f"Research paper excerpts:\n\n{context}\n\n"
            f"Question: {question}"
        )

        try:
            import requests
            response = requests.post(
                self.ANTHROPIC_API_URL,
                json={
                    "model": self.MODEL,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"Error calling LLM: {e}\n\nContext preview:\n{context[:500]}…"

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] From '{chunk.paper_title}' (relevance: {chunk.score:.3f}):\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _get_faiss_index(self) -> "faiss.Index":
        if self._index is None:
            dim = self._embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self._embeddings.astype(np.float32))
            self._index = index
        return self._index
