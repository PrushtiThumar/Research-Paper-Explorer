"""
Research Paper Semantic Knowledge Explorer
==========================================
Streamlit dashboard — main entry point.

Run with:
    streamlit run src/dashboard.py
"""

import io
import json
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

# Allow imports from src/ when running as a script
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B263B 100%);
    border-right: 1px solid rgba(74,144,217,0.2);
}
[data-testid="stSidebar"] * { color: #E0E8F0 !important; }

/* Main background */
.stApp { background-color: #0E1117; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1B263B, #0D1B2A);
    border: 1px solid rgba(74,144,217,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #4A90D9; }
.metric-card .label { font-size: 0.8rem; color: #8899AA; text-transform: uppercase; letter-spacing: 1px; }

/* Paper card */
.paper-card {
    background: #1B263B;
    border: 1px solid rgba(74,144,217,0.2);
    border-left: 4px solid #4A90D9;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.paper-card .ptitle { font-weight: 600; font-size: 1rem; color: #E0E8F0; }
.paper-card .pmeta { font-size: 0.8rem; color: #8899AA; margin-top: 4px; }
.paper-card .pkeywords { font-size: 0.75rem; color: #F5A623; margin-top: 6px; }

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #0D2137, #0D1B2A);
    border: 1px solid rgba(74,144,217,0.4);
    border-radius: 10px;
    padding: 20px;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #D0E0F0;
}

/* Tabs */
button[data-baseweb="tab"] { font-family: 'Space Grotesk', sans-serif !important; }

/* Keyword badge */
.kw-badge {
    display: inline-block;
    background: rgba(245,166,35,0.15);
    border: 1px solid rgba(245,166,35,0.4);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #F5A623;
    margin: 2px;
}

/* Status bar */
.status-bar {
    background: #1B263B;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #8899AA;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "papers": {},          # paper_id → dict with meta + chunks + embedding
        "sim_engine": None,
        "graph_builder": None,
        "rag_engine": None,
        "emb_gen": None,
        "kw_extractor": None,
        "viz_manager": None,
        "keyword_freq": {},    # keyword → count
        "loading_models": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Lazy component loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_components():
    """Load heavy ML components once and cache across sessions."""
    from embedding_generator import EmbeddingGenerator
    from similarity_engine import SimilarityEngine
    from graph_builder import KnowledgeGraphBuilder
    from rag_engine import RAGEngine
    from keyword_extractor import KeywordExtractor
    from visualization_manager import VisualizationManager

    emb_gen = EmbeddingGenerator()
    sim_engine = SimilarityEngine(embedding_dim=emb_gen.embedding_dim)
    graph = KnowledgeGraphBuilder()
    rag = RAGEngine(embedding_generator=emb_gen)
    kw_ext = KeywordExtractor()
    viz = VisualizationManager()
    return emb_gen, sim_engine, graph, rag, kw_ext, viz


def get_components():
    try:
        emb_gen, sim_engine, graph, rag, kw_ext, viz = _load_components()
        st.session_state.emb_gen = emb_gen
        st.session_state.sim_engine = sim_engine
        st.session_state.graph_builder = graph
        st.session_state.rag_engine = rag
        st.session_state.kw_extractor = kw_ext
        st.session_state.viz_manager = viz
        return True
    except Exception as e:
        st.error(f"Failed to load ML components: {e}")
        return False


# ── Paper processing ──────────────────────────────────────────────────────────
def process_paper(pdf_bytes: bytes, source_name: str) -> Optional[str]:
    """Full pipeline: extract → metadata → keywords → embed → index."""
    from pdf_processor import PDFProcessor
    from metadata_extractor import MetadataExtractor

    emb_gen = st.session_state.emb_gen
    sim_engine = st.session_state.sim_engine
    graph = st.session_state.graph_builder
    rag = st.session_state.rag_engine
    kw_ext = st.session_state.kw_extractor

    progress = st.progress(0, text="Extracting text…")
    try:
        # 1. Extract
        proc = PDFProcessor(chunk_size=500, chunk_overlap=50)
        doc = proc.process_bytes(pdf_bytes, source=source_name)
        progress.progress(20, text="Extracting metadata…")

        # 2. Metadata
        meta_ext = MetadataExtractor()
        meta = meta_ext.extract(doc.cleaned_text, source=source_name)
        progress.progress(40, text="Extracting keywords…")

        # 3. Keywords
        keywords = kw_ext.extract_from_chunks(doc.chunks[:10])
        if meta.keywords:
            keywords = list(dict.fromkeys(meta.keywords + keywords))[:15]
        progress.progress(55, text="Generating embeddings…")

        # 4. Embed chunks & document
        chunk_embs = emb_gen.embed_batch(doc.chunks, show_progress=False)
        doc_emb = chunk_embs.mean(axis=0)
        import numpy as np
        n = np.linalg.norm(doc_emb)
        if n > 0:
            doc_emb /= n

        progress.progress(70, text="Indexing for search…")

        # 5. Create paper record
        paper_id = str(uuid.uuid4())[:8]
        paper_record = {
            "id": paper_id,
            "title": meta.title,
            "authors": meta.authors,
            "year": meta.year,
            "abstract": meta.abstract,
            "keywords": keywords,
            "source": source_name,
            "chunks": doc.chunks,
            "embedding": doc_emb,
            "num_pages": doc.num_pages,
        }
        st.session_state.papers[paper_id] = paper_record

        # 6. Similarity engine
        sim_engine.add_paper(paper_id, doc_emb, title=meta.title)
        progress.progress(80, text="Building knowledge graph…")

        # 7. Graph
        graph.add_paper(
            paper_id=paper_id,
            title=meta.title,
            authors=meta.authors,
            year=meta.year,
            abstract=meta.abstract,
            keywords=keywords,
        )

        # Recompute similarity edges for all pairs
        pids = sim_engine.paper_ids()
        if len(pids) > 1:
            sim_mat = sim_engine.similarity_matrix()
            for i, pid_a in enumerate(pids):
                for j, pid_b in enumerate(pids):
                    if j <= i:
                        continue
                    graph.add_similarity_edge(pid_a, pid_b, float(sim_mat[i, j]))

        progress.progress(90, text="Indexing chunks for Q&A…")

        # 8. RAG
        rag.add_chunks(doc.chunks, paper_id, meta.title, embeddings=chunk_embs)

        # 9. Keyword frequency
        for kw in keywords:
            st.session_state.keyword_freq[kw] = st.session_state.keyword_freq.get(kw, 0) + 1

        progress.progress(100, text="Done!")
        return paper_id

    except Exception as e:
        st.error(f"Failed to process paper: {e}")
        logger.exception(e)
        return None
    finally:
        progress.empty()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 🔬 Research Explorer")
        st.markdown("---")

        # Model status
        components_ok = st.session_state.emb_gen is not None
        if not components_ok:
            with st.spinner("Loading ML models…"):
                components_ok = get_components()

        if components_ok:
            st.success("✅ Models loaded", icon="🤖")
        else:
            st.error("❌ Model loading failed")
            return

        st.markdown("---")
        st.markdown("### 📥 Add Papers")

        # Upload
        uploaded = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        # arXiv URL
        arxiv_url = st.text_input("Or paste arXiv URL / ID", placeholder="https://arxiv.org/abs/…")

        if st.button("🚀 Process Papers", use_container_width=True):
            files_to_process = []

            for f in (uploaded or []):
                # Avoid reprocessing
                already = any(
                    p["source"] == f.name for p in st.session_state.papers.values()
                )
                if not already:
                    files_to_process.append((f.read(), f.name))

            if arxiv_url.strip():
                try:
                    from pdf_processor import PDFProcessor
                    with st.spinner("Downloading from arXiv…"):
                        proc = PDFProcessor()
                        doc_tmp = proc.process_url(arxiv_url.strip())
                        files_to_process.append((
                            _url_to_bytes(arxiv_url.strip()),
                            arxiv_url.strip(),
                        ))
                except Exception as e:
                    st.error(f"arXiv download failed: {e}")

            if not files_to_process:
                st.info("No new papers to process.")
            else:
                for pdf_bytes, name in files_to_process:
                    st.markdown(f"**Processing:** `{name[:40]}`")
                    pid = process_paper(pdf_bytes, name)
                    if pid:
                        title = st.session_state.papers[pid]["title"]
                        st.success(f"✓ {title[:50]}")

        st.markdown("---")
        # Stats
        n = len(st.session_state.papers)
        st.markdown(f"**Papers loaded:** {n}")
        n_kw = len(st.session_state.keyword_freq)
        st.markdown(f"**Unique keywords:** {n_kw}")

        if n > 0 and st.button("🗑 Clear all papers", use_container_width=True):
            _reset_state()
            st.rerun()


def _url_to_bytes(url: str) -> bytes:
    """Download PDF bytes from URL."""
    import requests
    resp = requests.get(url if url.endswith('.pdf') else url.replace('/abs/', '/pdf/') + '.pdf', timeout=30)
    resp.raise_for_status()
    return resp.content


def _reset_state():
    for key in ["papers", "keyword_freq"]:
        st.session_state[key] = {} if key != "keyword_freq" else {}
    st.cache_resource.clear()
    _init_state()


# ── Main tabs ─────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        "<h1 style='color:#4A90D9;margin-bottom:0'>🔬 Research Paper Semantic Knowledge Explorer</h1>"
        "<p style='color:#8899AA;font-size:0.95rem;margin-top:4px'>Upload papers → explore concepts → ask questions</p>",
        unsafe_allow_html=True,
    )

    sidebar()

    papers = st.session_state.papers
    n_papers = len(papers)

    # ── Metrics row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='value'>{n_papers}</div><div class='label'>Papers</div></div>", unsafe_allow_html=True)
    with c2:
        n_kw = len(st.session_state.keyword_freq)
        st.markdown(f"<div class='metric-card'><div class='value'>{n_kw}</div><div class='label'>Keywords</div></div>", unsafe_allow_html=True)
    with c3:
        total_chunks = sum(len(p.get("chunks", [])) for p in papers.values())
        st.markdown(f"<div class='metric-card'><div class='value'>{total_chunks}</div><div class='label'>Chunks Indexed</div></div>", unsafe_allow_html=True)
    with c4:
        graph = st.session_state.graph_builder
        n_edges = graph.graph.number_of_edges() if graph else 0
        st.markdown(f"<div class='metric-card'><div class='value'>{n_edges}</div><div class='label'>Graph Edges</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if n_papers == 0:
        st.info("👈 Upload PDF papers from the sidebar to get started.", icon="📄")
        _show_demo_info()
        return

    tab_library, tab_search, tab_similarity, tab_graph, tab_qa = st.tabs([
        "📚 Library",
        "🔍 Search",
        "📊 Similarity",
        "🌐 Knowledge Graph",
        "💬 Ask a Question",
    ])

    with tab_library:
        _tab_library(papers)

    with tab_search:
        _tab_search(papers)

    with tab_similarity:
        _tab_similarity(papers)

    with tab_graph:
        _tab_graph()

    with tab_qa:
        _tab_qa()


# ── Tab: Library ──────────────────────────────────────────────────────────────
def _tab_library(papers: dict):
    st.subheader("Loaded Papers")

    for pid, p in papers.items():
        kw_html = " ".join(f"<span class='kw-badge'>{k}</span>" for k in p.get("keywords", [])[:8])
        authors_str = ", ".join(p.get("authors", [])[:3]) or "Unknown"
        year_str = str(p.get("year", "")) or "?"
        st.markdown(
            f"<div class='paper-card'>"
            f"<div class='ptitle'>{p['title']}</div>"
            f"<div class='pmeta'>👥 {authors_str} &nbsp;|&nbsp; 📅 {year_str} &nbsp;|&nbsp; 📄 {p.get('num_pages','?')} pages</div>"
            f"<div class='pkeywords'>{kw_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if p.get("abstract"):
            with st.expander("Abstract", expanded=False):
                st.write(p["abstract"])

    # Keyword frequency chart
    if st.session_state.keyword_freq:
        st.subheader("Keyword Distribution")
        viz = st.session_state.viz_manager
        if viz:
            fig = viz.keyword_frequency_chart(st.session_state.keyword_freq, top_n=20)
            st.plotly_chart(fig, use_container_width=True)


# ── Tab: Search ───────────────────────────────────────────────────────────────
def _tab_search(papers: dict):
    st.subheader("Search by Keyword or Topic")

    query = st.text_input("Search papers", placeholder="e.g. attention mechanism, diffusion models…")
    if not query:
        return

    query_lower = query.lower()
    results = []

    for pid, p in papers.items():
        score = 0
        kws = [k.lower() for k in p.get("keywords", [])]
        title_lower = p["title"].lower()
        abstract_lower = p.get("abstract", "").lower()

        # Exact keyword match
        for kw in kws:
            if query_lower in kw or kw in query_lower:
                score += 3
        # Title match
        if query_lower in title_lower:
            score += 2
        # Abstract match
        if query_lower in abstract_lower:
            score += 1

        if score > 0:
            results.append((pid, p, score))

    # Semantic fallback
    if not results and st.session_state.emb_gen and st.session_state.sim_engine:
        st.info("No exact matches — running semantic search…")
        q_emb = st.session_state.emb_gen.embed_text(query)
        sim_results = st.session_state.sim_engine.search_by_embedding(q_emb, top_k=5, min_score=0.2)
        for r in sim_results:
            if r.paper_id in papers:
                results.append((r.paper_id, papers[r.paper_id], r.score * 10))

    results.sort(key=lambda x: -x[2])

    if results:
        st.success(f"Found {len(results)} result(s).")
        for pid, p, score in results:
            kw_html = " ".join(f"<span class='kw-badge'>{k}</span>" for k in p.get("keywords", [])[:8])
            st.markdown(
                f"<div class='paper-card'>"
                f"<div class='ptitle'>{p['title']}</div>"
                f"<div class='pmeta'>Relevance score: {score}</div>"
                f"<div class='pkeywords'>{kw_html}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("No results found.")


# ── Tab: Similarity ───────────────────────────────────────────────────────────
def _tab_similarity(papers: dict):
    st.subheader("Paper Similarity Explorer")
    sim_engine = st.session_state.sim_engine
    viz = st.session_state.viz_manager

    if not sim_engine or sim_engine.paper_count() < 2:
        st.info("Upload at least 2 papers to see similarity analysis.")
        return

    # Heatmap
    st.markdown("#### Similarity Matrix")
    pids = sim_engine.paper_ids()
    titles = [papers[pid]["title"] for pid in pids if pid in papers]
    import numpy as np
    sim_mat = sim_engine.similarity_matrix()
    fig = viz.similarity_heatmap(sim_mat, pids, titles)
    st.plotly_chart(fig, use_container_width=True)

    # Per-paper similarity
    st.markdown("#### Most Similar Papers")
    paper_options = {p["title"]: pid for pid, p in papers.items()}
    selected_title = st.selectbox("Select a paper", list(paper_options.keys()))
    selected_pid = paper_options[selected_title]

    sims = sim_engine.most_similar(selected_pid, top_k=10)
    if sims:
        sim_pairs = [(papers[r.paper_id]["title"] if r.paper_id in papers else r.paper_id, r.score) for r in sims]
        fig2 = viz.paper_similarity_bar(sim_pairs, selected_title)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Top matches:**")
        for r in sims[:5]:
            if r.paper_id in papers:
                p = papers[r.paper_id]
                st.markdown(f"- **{p['title']}** — similarity: `{r.score:.4f}`")
    else:
        st.info("No similar papers found above threshold.")


# ── Tab: Knowledge Graph ──────────────────────────────────────────────────────
def _tab_graph():
    st.subheader("Interactive Knowledge Graph")
    graph = st.session_state.graph_builder
    viz = st.session_state.viz_manager

    if not graph or graph.graph.number_of_nodes() == 0:
        st.info("No graph data yet. Upload papers to build the knowledge graph.")
        return

    summary = graph.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", summary["total_nodes"])
    c2.metric("Papers", summary["papers"])
    c3.metric("Keywords", summary["keywords"])
    c4.metric("Edges", summary["total_edges"])

    view_mode = st.radio("Graph engine", ["Plotly (fast)", "PyVis (interactive)"], horizontal=True)

    if view_mode == "Plotly (fast)":
        graph_data = graph.to_plotly_data()
        fig = viz.graph_plotly(graph_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        html = graph.to_pyvis_html(height="600px")
        if html:
            st.components.v1.html(html, height=620, scrolling=False)
        else:
            st.warning("PyVis not installed. Install with: `pip install pyvis`")

    # Keyword lookup
    st.markdown("#### Find papers by keyword")
    all_kws = sorted(graph.keywords())
    if all_kws:
        kw_sel = st.selectbox("Keyword", all_kws)
        matching = graph.papers_with_keyword(kw_sel)
        if matching:
            papers_state = st.session_state.papers
            for pid in matching:
                title = papers_state.get(pid, {}).get("title", pid)
                st.markdown(f"- 📄 **{title}**")
        else:
            st.info("No papers found for that keyword.")


# ── Tab: Q&A ──────────────────────────────────────────────────────────────────
def _tab_qa():
    st.subheader("Ask Questions About Your Papers")
    rag = st.session_state.rag_engine

    if not rag or rag.chunk_count() == 0:
        st.info("Upload papers first to enable question answering.")
        return

    st.markdown(f"*{rag.chunk_count()} chunks indexed across {len(st.session_state.papers)} paper(s).*")

    # API key input
    api_key = st.text_input(
        "Anthropic API Key (for LLM answers)",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required for AI-generated answers. Set ANTHROPIC_API_KEY env var to avoid re-entering.",
    )
    if api_key:
        rag.api_key = api_key

    st.markdown("---")
    question = st.text_area(
        "Your question",
        placeholder="Which papers discuss transformers? What methods are used for image segmentation?",
        height=100,
    )

    top_k = st.slider("Chunks to retrieve", 3, 15, 5)

    if st.button("🔎 Get Answer", use_container_width=True) and question.strip():
        with st.spinner("Searching and generating answer…"):
            answer = rag.ask(question, top_k=top_k)

        st.markdown("#### Answer")
        st.markdown(f"<div class='answer-box'>{answer.answer}</div>", unsafe_allow_html=True)

        st.markdown("#### Sources Used")
        seen: set[str] = set()
        for src in answer.sources:
            if src.paper_title not in seen:
                seen.add(src.paper_title)
                st.markdown(
                    f"- 📄 **{src.paper_title}** &nbsp; `relevance: {src.score:.3f}`"
                )

        with st.expander("View retrieved context"):
            for i, src in enumerate(answer.sources, 1):
                st.markdown(f"**Chunk {i}** — {src.paper_title} (score: {src.score:.3f})")
                st.text(src.text[:400] + ("…" if len(src.text) > 400 else ""))
                st.markdown("---")


# ── Demo info (no papers loaded) ──────────────────────────────────────────────
def _show_demo_info():
    st.markdown("---")
    st.markdown("### How it works")
    cols = st.columns(3)
    steps = [
        ("📥 Upload", "Add PDF papers or arXiv links via the sidebar. The system extracts text, metadata, and keywords automatically."),
        ("🧠 Analyse", "Papers are embedded using Sentence Transformers. Semantic similarity is computed and a knowledge graph is built."),
        ("💬 Explore", "Search by keyword, explore the graph, compare similarities, or ask natural language questions using RAG."),
    ]
    for col, (icon_title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div class='paper-card'><div class='ptitle'>{icon_title}</div>"
                f"<div class='pmeta'>{desc}</div></div>",
                unsafe_allow_html=True,
            )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
