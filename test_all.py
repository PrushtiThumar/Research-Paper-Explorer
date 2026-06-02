"""
Unit tests for the Research Paper Semantic Knowledge Explorer.

Run with:
    pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# PDF Processor
# ─────────────────────────────────────────────────────────────────────────────
class TestPDFProcessor:
    def setup_method(self):
        from pdf_processor import PDFProcessor
        self.proc = PDFProcessor(chunk_size=100, chunk_overlap=10)

    def test_clean_removes_extra_spaces(self):
        text = "hello   world\n\n\n\nfoo"
        cleaned = self.proc._clean(text)
        assert "   " not in cleaned

    def test_clean_removes_references_section(self):
        text = "Introduction text\n\nReferences\n\nSmith, J. (2020)..."
        cleaned = self.proc._clean(text)
        assert "Smith" not in cleaned

    def test_chunk_produces_correct_count(self):
        words = " ".join(["word"] * 250)
        chunks = self.proc._chunk(words)
        # 250 words, chunk=100, overlap=10 → ceil((250-100)/(100-10)) + 1 ≈ 3 chunks
        assert len(chunks) >= 2

    def test_chunk_overlap(self):
        words = " ".join([str(i) for i in range(150)])
        chunks = self.proc._chunk(words)
        # Last words of chunk[0] should appear at start of chunk[1]
        end_of_first = set(chunks[0].split()[-10:])
        start_of_second = set(chunks[1].split()[:10])
        assert end_of_first & start_of_second  # overlap exists

    def test_arxiv_url_resolution(self):
        url = "https://arxiv.org/abs/2301.00001"
        resolved = self.proc._resolve_arxiv_url(url)
        assert "pdf" in resolved and "2301.00001" in resolved

    def test_arxiv_bare_id(self):
        resolved = self.proc._resolve_arxiv_url("2301.00001")
        assert resolved.endswith(".pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Extractor
# ─────────────────────────────────────────────────────────────────────────────
class TestMetadataExtractor:
    def setup_method(self):
        from metadata_extractor import MetadataExtractor
        self.ext = MetadataExtractor()

    def test_extracts_year(self):
        text = "Published in 2023. This paper presents..."
        meta = self.ext.extract(text)
        assert meta.year == 2023

    def test_no_year_returns_none(self):
        text = "This paper presents a new approach."
        meta = self.ext.extract(text)
        assert meta.year is None

    def test_extracts_abstract(self):
        text = "Title Here\n\nAbstract\nWe propose a novel method.\n\n1. Introduction\nMore text."
        meta = self.ext.extract(text)
        assert "novel method" in meta.abstract

    def test_extracts_keywords(self):
        text = "Paper Title\n\nAbstract text.\n\nKeywords: deep learning, transformers, NLP\n\nIntroduction"
        meta = self.ext.extract(text)
        assert len(meta.keywords) >= 1

    def test_title_not_empty(self):
        text = "Attention Is All You Need\n\nAbstract\nWe present..."
        meta = self.ext.extract(text)
        assert meta.title and meta.title != "Unknown Title"

    def test_metadata_to_dict(self):
        from metadata_extractor import PaperMetadata
        m = PaperMetadata(title="Test", authors=["A"], year=2020)
        d = m.to_dict()
        assert d["title"] == "Test"
        assert d["year"] == 2020


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Extractor
# ─────────────────────────────────────────────────────────────────────────────
class TestKeywordExtractor:
    def setup_method(self):
        from keyword_extractor import KeywordExtractor
        # Force TF-IDF (no KeyBERT needed)
        self.ext = KeywordExtractor(use_keybert=False)

    def test_extracts_keywords(self):
        text = (
            "Neural networks are powerful machine learning models used in deep learning. "
            "Convolutional neural networks excel at image classification tasks. "
            "Recurrent neural networks handle sequential data."
        )
        kws = self.ext.extract(text)
        assert len(kws) > 0

    def test_no_stopwords_in_results(self):
        from keyword_extractor import _STOPWORDS
        text = "The quick brown fox jumps over the lazy dog and the cat."
        kws = self.ext.extract(text)
        for kw in kws:
            for word in kw.split():
                assert word not in _STOPWORDS, f"Stopword '{word}' found in keyword '{kw}'"

    def test_extract_from_chunks(self):
        chunks = [
            "Transformer architecture uses self-attention mechanisms.",
            "BERT is a bidirectional transformer for NLP tasks.",
            "GPT models generate text using transformer layers.",
        ]
        kws = self.ext.extract_from_chunks(chunks)
        assert isinstance(kws, list)
        assert len(kws) > 0

    def test_empty_text_returns_empty(self):
        kws = self.ext.extract("")
        assert kws == []


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Engine
# ─────────────────────────────────────────────────────────────────────────────
class TestSimilarityEngine:
    def setup_method(self):
        from similarity_engine import SimilarityEngine
        self.engine = SimilarityEngine(embedding_dim=4)

    def _vec(self, *vals):
        v = np.array(vals, dtype=np.float32)
        return v / np.linalg.norm(v)

    def test_add_and_retrieve(self):
        self.engine.add_paper("p1", self._vec(1, 0, 0, 0), "Paper One")
        self.engine.add_paper("p2", self._vec(0, 1, 0, 0), "Paper Two")
        assert self.engine.paper_count() == 2

    def test_identical_papers_score_one(self):
        v = self._vec(1, 1, 1, 1)
        self.engine.add_paper("pa", v, "A")
        self.engine.add_paper("pb", v.copy(), "B")
        score = self.engine.paper_similarity("pa", "pb")
        assert abs(score - 1.0) < 1e-5

    def test_orthogonal_papers_score_zero(self):
        self.engine.add_paper("px", self._vec(1, 0, 0, 0), "X")
        self.engine.add_paper("py", self._vec(0, 1, 0, 0), "Y")
        score = self.engine.paper_similarity("px", "py")
        assert abs(score) < 1e-5

    def test_most_similar_excludes_self(self):
        for i in range(4):
            self.engine.add_paper(f"p{i}", self._vec(*np.random.rand(4)), f"Paper {i}")
        results = self.engine.most_similar("p0")
        ids = [r.paper_id for r in results]
        assert "p0" not in ids

    def test_similarity_matrix_shape(self):
        for i in range(3):
            self.engine.add_paper(f"m{i}", self._vec(*np.random.rand(4)))
        mat = self.engine.similarity_matrix()
        assert mat.shape == (3, 3)

    def test_missing_paper_raises(self):
        with pytest.raises(ValueError):
            self.engine.paper_similarity("nonexistent_a", "nonexistent_b")


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph Builder
# ─────────────────────────────────────────────────────────────────────────────
class TestKnowledgeGraphBuilder:
    def setup_method(self):
        from graph_builder import KnowledgeGraphBuilder
        self.gb = KnowledgeGraphBuilder()

    def test_add_paper_creates_node(self):
        self.gb.add_paper("p1", "Test Paper", keywords=["ml", "ai"])
        assert "p1" in self.gb.graph.nodes

    def test_keywords_become_nodes(self):
        self.gb.add_paper("p1", "Test", keywords=["transformer", "nlp"])
        assert "transformer" in self.gb.graph.nodes
        assert "nlp" in self.gb.graph.nodes

    def test_similarity_edge_above_threshold(self):
        self.gb.add_paper("p1", "Paper 1")
        self.gb.add_paper("p2", "Paper 2")
        self.gb.add_similarity_edge("p1", "p2", 0.85)
        assert self.gb.graph.has_edge("p1", "p2")

    def test_similarity_edge_below_threshold_not_added(self):
        self.gb.add_paper("p1", "Paper 1")
        self.gb.add_paper("p2", "Paper 2")
        self.gb.add_similarity_edge("p1", "p2", 0.1)
        assert not self.gb.graph.has_edge("p1", "p2")

    def test_papers_with_keyword(self):
        self.gb.add_paper("p1", "Paper 1", keywords=["attention"])
        self.gb.add_paper("p2", "Paper 2", keywords=["attention", "bert"])
        result = self.gb.papers_with_keyword("attention")
        assert "p1" in result
        assert "p2" in result

    def test_summary_counts(self):
        self.gb.add_paper("pa", "A", keywords=["kw1", "kw2"])
        self.gb.add_paper("pb", "B", keywords=["kw1"])
        s = self.gb.summary()
        assert s["papers"] == 2
        assert s["keywords"] >= 2

    def test_plotly_data_structure(self):
        self.gb.add_paper("p1", "Paper One", keywords=["ml"])
        data = self.gb.to_plotly_data()
        assert "nodes" in data and "edges" in data
        assert len(data["nodes"]) >= 2  # paper + keyword


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────
class TestRAGEngine:
    """Tests that don't require a real LLM call."""

    class MockEmbedGen:
        embedding_dim = 4

        def embed_text(self, text):
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            v = rng.rand(4).astype(np.float32)
            return v / np.linalg.norm(v)

        def embed_batch(self, texts, **kw):
            return np.array([self.embed_text(t) for t in texts])

    def setup_method(self):
        from rag_engine import RAGEngine
        self.rag = RAGEngine(embedding_generator=self.MockEmbedGen(), api_key="")

    def test_add_and_count_chunks(self):
        self.rag.add_chunks(["chunk one", "chunk two", "chunk three"], "p1", "Paper One")
        assert self.rag.chunk_count() == 3

    def test_retrieve_returns_results(self):
        self.rag.add_chunks(["transformers are great", "attention mechanism"], "p1", "Paper One")
        results = self.rag.retrieve("transformer architecture", top_k=2)
        assert len(results) > 0
        assert results[0].paper_title == "Paper One"

    def test_remove_paper(self):
        self.rag.add_chunks(["data chunk"], "p1", "Paper One")
        self.rag.add_chunks(["other chunk"], "p2", "Paper Two")
        self.rag.remove_paper("p1")
        assert self.rag.chunk_count() == 1

    def test_ask_no_api_key_returns_graceful(self):
        self.rag.add_chunks(["relevant context about transformers"], "p1", "Paper One")
        ans = self.rag.ask("Tell me about transformers")
        assert isinstance(ans.answer, str)
        assert len(ans.sources) > 0

    def test_ask_empty_index(self):
        ans = self.rag.ask("Anything?")
        assert "No relevant" in ans.answer
