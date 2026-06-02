"""
KeywordExtractor: Extracts important concepts and research topics from paper text.
Uses KeyBERT when available, falls back to TF-IDF.
"""

import re
import math
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    logger.info("KeyBERT not available, using TF-IDF fallback.")

# Common English stopwords (lightweight, no NLTK required at import time)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "we", "our", "us",
    "they", "their", "them", "he", "she", "his", "her", "you", "your",
    "i", "me", "my", "not", "no", "so", "as", "if", "than", "then",
    "also", "paper", "show", "using", "used", "based", "approach",
    "method", "results", "result", "section", "figure", "table",
    "proposed", "model", "models", "data", "set", "number", "two",
    "one", "three", "new", "use", "while", "which", "such", "each",
    "both", "between", "however", "therefore", "thus", "hence",
}


class KeywordExtractor:
    """
    Extracts keyphrases representing concepts, topics, and methods.

    Strategy:
    1. KeyBERT (transformer-backed) if installed.
    2. TF-IDF unigram/bigram fallback otherwise.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_n: int = 15,
        use_keybert: bool = True,
    ):
        self.top_n = top_n
        self._kw_model: Optional[object] = None

        if use_keybert and KEYBERT_AVAILABLE:
            try:
                self._kw_model = KeyBERT(model=model_name)
                logger.info("KeyBERT initialised with model: %s", model_name)
            except Exception as e:
                logger.warning("KeyBERT init failed (%s); using TF-IDF.", e)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        text: str,
        corpus: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Extract keywords from text.

        Args:
            text: Document text (or a concatenation of chunks).
            corpus: Other documents in the collection (improves TF-IDF IDF).

        Returns:
            Sorted list of keyword strings.
        """
        if not text.strip():
            return []

        if self._kw_model is not None:
            return self._keybert_extract(text)
        return self._tfidf_extract(text, corpus or [])

    def extract_from_chunks(
        self,
        chunks: list[str],
        corpus: Optional[list[str]] = None,
        top_n_per_chunk: int = 5,
    ) -> list[str]:
        """
        Extract keywords from multiple chunks and merge results.

        Useful when the full text is too long for KeyBERT in one pass.
        """
        combined_counts: Counter = Counter()

        for chunk in chunks:
            kws = self.extract(chunk, corpus)
            combined_counts.update(kws)

        # Return the most frequently occurring keywords across chunks
        return [kw for kw, _ in combined_counts.most_common(self.top_n)]

    # ------------------------------------------------------------------ #
    #  KeyBERT                                                            #
    # ------------------------------------------------------------------ #

    def _keybert_extract(self, text: str) -> list[str]:
        # Truncate to 512 tokens worth (~2 500 chars) to stay within model limits
        text = text[:2500]
        try:
            results = self._kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=self.top_n,
                use_mmr=True,
                diversity=0.5,
            )
            return [kw for kw, _score in results]
        except Exception as e:
            logger.warning("KeyBERT extraction failed: %s", e)
            return self._tfidf_extract(text, [])

    # ------------------------------------------------------------------ #
    #  TF-IDF fallback                                                    #
    # ------------------------------------------------------------------ #

    def _tfidf_extract(self, text: str, corpus: list[str]) -> list[str]:
        """Simple TF-IDF implementation (no sklearn required)."""
        doc_tokens = self._tokenise(text)
        doc_tf = self._term_freq(doc_tokens)

        all_docs = [text] + corpus
        num_docs = len(all_docs)
        idf: dict[str, float] = {}
        for term in doc_tf:
            df = sum(1 for d in all_docs if term in self._tokenise(d))
            idf[term] = math.log((num_docs + 1) / (df + 1)) + 1.0

        scores = {term: tf * idf.get(term, 1.0) for term, tf in doc_tf.items()}

        # Also score bigrams
        bigrams = self._bigrams(doc_tokens)
        for bg, count in bigrams.items():
            if bg not in _STOPWORDS:
                tf_bg = count / max(len(doc_tokens) - 1, 1)
                w0, w1 = bg.split()
                avg_idf = (idf.get(w0, 1.0) + idf.get(w1, 1.0)) / 2
                scores[bg] = tf_bg * avg_idf * 1.2  # slight boost for phrases

        # Sort and filter
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        seen_words: set[str] = set()
        results: list[str] = []
        for term, _score in ranked:
            words = term.split()
            # Skip if any word already covered
            if any(w in seen_words for w in words):
                continue
            results.append(term)
            seen_words.update(words)
            if len(results) >= self.top_n:
                break

        return results

    def _tokenise(self, text: str) -> list[str]:
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z-]{2,}\b', text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    def _term_freq(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {t: c / total for t, c in counts.items()}

    def _bigrams(self, tokens: list[str]) -> Counter:
        return Counter(f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1))
