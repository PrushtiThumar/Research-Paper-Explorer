"""
SimilarityEngine: Computes cosine similarity between papers and concepts,
and builds a FAISS index for fast approximate nearest-neighbour search.
"""

import logging
from typing import Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not available; falling back to brute-force cosine search.")


@dataclass
class SimilarityResult:
    """A single similarity match."""
    paper_id: str
    score: float
    title: str = ""

    def __repr__(self) -> str:
        return f"SimilarityResult(id={self.paper_id!r}, score={self.score:.4f}, title={self.title!r})"


class SimilarityEngine:
    """
    Manages a collection of paper embeddings and supports:
    - Paper-to-paper similarity (cosine)
    - Query-to-paper search (nearest neighbours)
    - Concept-level similarity

    Papers are identified by string IDs (e.g. filename or UUID).
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._ids: list[str] = []
        self._titles: dict[str, str] = {}
        self._matrix: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        self._index: object = None  # FAISS index (lazy)

    # ------------------------------------------------------------------ #
    #  Adding papers                                                       #
    # ------------------------------------------------------------------ #

    def add_paper(self, paper_id: str, embedding: np.ndarray, title: str = "") -> None:
        """Add a single paper embedding."""
        vec = self._normalise(embedding.astype(np.float32))
        self._ids.append(paper_id)
        self._titles[paper_id] = title
        self._matrix = (
            np.vstack([self._matrix, vec[np.newaxis, :]])
            if self._matrix.size
            else vec[np.newaxis, :]
        )
        self._index = None  # Invalidate FAISS index

    def add_papers(
        self,
        paper_ids: list[str],
        embeddings: np.ndarray,
        titles: Optional[list[str]] = None,
    ) -> None:
        """Bulk-add paper embeddings."""
        titles = titles or [""] * len(paper_ids)
        for pid, emb, title in zip(paper_ids, embeddings, titles):
            self.add_paper(pid, emb, title)

    def update_paper(self, paper_id: str, embedding: np.ndarray, title: str = "") -> None:
        """Update or insert a paper."""
        if paper_id in self._ids:
            idx = self._ids.index(paper_id)
            self._matrix[idx] = self._normalise(embedding.astype(np.float32))
            self._titles[paper_id] = title
            self._index = None
        else:
            self.add_paper(paper_id, embedding, title)

    # ------------------------------------------------------------------ #
    #  Similarity queries                                                  #
    # ------------------------------------------------------------------ #

    def paper_similarity(self, id_a: str, id_b: str) -> float:
        """Cosine similarity between two stored papers."""
        if id_a not in self._ids or id_b not in self._ids:
            raise ValueError(f"Unknown paper id: {id_a!r} or {id_b!r}")
        va = self._matrix[self._ids.index(id_a)]
        vb = self._matrix[self._ids.index(id_b)]
        return float(np.dot(va, vb))  # Already normalised → dot == cosine

    def most_similar(
        self,
        paper_id: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SimilarityResult]:
        """Return the top-k most similar papers to a given paper."""
        if paper_id not in self._ids:
            raise ValueError(f"Unknown paper id: {paper_id!r}")

        query_vec = self._matrix[self._ids.index(paper_id)]
        return self._search(query_vec, top_k=top_k + 1, min_score=min_score, exclude_id=paper_id)

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SimilarityResult]:
        """Return the top-k papers closest to an arbitrary query embedding."""
        query_vec = self._normalise(query_embedding.astype(np.float32))
        return self._search(query_vec, top_k=top_k, min_score=min_score)

    def similarity_matrix(self) -> np.ndarray:
        """Return full N×N pairwise cosine similarity matrix."""
        if self._matrix.size == 0:
            return np.empty((0, 0))
        return (self._matrix @ self._matrix.T).clip(-1, 1)

    def paper_ids(self) -> list[str]:
        return list(self._ids)

    def paper_count(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        min_score: float,
        exclude_id: Optional[str] = None,
    ) -> list[SimilarityResult]:
        if self._matrix.size == 0:
            return []

        if FAISS_AVAILABLE:
            results = self._faiss_search(query_vec, top_k)
        else:
            scores = (self._matrix @ query_vec).clip(-1, 1)
            top_indices = np.argsort(-scores)[:top_k]
            results = [
                SimilarityResult(
                    paper_id=self._ids[i],
                    score=float(scores[i]),
                    title=self._titles.get(self._ids[i], ""),
                )
                for i in top_indices
            ]

        # Filter
        results = [r for r in results if r.score >= min_score]
        if exclude_id:
            results = [r for r in results if r.paper_id != exclude_id]
        return results

    def _faiss_search(self, query_vec: np.ndarray, top_k: int) -> list[SimilarityResult]:
        index = self._get_faiss_index()
        k = min(top_k, self.paper_count())
        dists, idxs = index.search(query_vec[np.newaxis, :], k)
        return [
            SimilarityResult(
                paper_id=self._ids[i],
                score=float(d),
                title=self._titles.get(self._ids[i], ""),
            )
            for d, i in zip(dists[0], idxs[0])
            if i >= 0
        ]

    def _get_faiss_index(self) -> "faiss.Index":
        if self._index is None:
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product on normalised vecs
            index.add(self._matrix)
            self._index = index
        return self._index

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec



