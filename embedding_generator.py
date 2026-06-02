"""
EmbeddingGenerator: Computes sentence-level embeddings for papers and chunks
using Sentence Transformers.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Embeddings disabled.")


class EmbeddingGenerator:
    """
    Wraps SentenceTransformer to produce L2-normalised embeddings.

    Default model: all-MiniLM-L6-v2 (fast, 384-dim, excellent for semantic search).
    Any SentenceTransformer-compatible model name is accepted.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: Optional[object] = None

        if not ST_AVAILABLE:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    # ------------------------------------------------------------------ #
    #  Lazy loading                                                        #
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns:
            1-D float32 numpy array (normalised).
        """
        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of strings.

        Returns:
            2-D float32 numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def embed_chunks(self, chunks: list[str]) -> np.ndarray:
        """
        Embed document chunks and return per-document aggregate embedding
        (mean-pooled, then re-normalised).
        """
        if not chunks:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        chunk_vecs = self.embed_batch(chunks)
        doc_vec = chunk_vecs.mean(axis=0)
        norm = np.linalg.norm(doc_vec)
        if norm > 0:
            doc_vec /= norm
        return doc_vec.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
