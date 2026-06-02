# 🔬 Research Paper Semantic Knowledge Explorer

An end-to-end system for processing, understanding, and interactively exploring research papers using NLP and semantic AI.

---

## Features

| Feature | Description |
|---|---|
| 📥 Ingest | Upload PDFs or paste arXiv URLs |
| 🔍 Metadata | Auto-extract title, authors, abstract, year, keywords |
| 🧠 Embeddings | Sentence-transformer embeddings (`all-MiniLM-L6-v2`) |
| 📊 Similarity | Cosine similarity matrix + FAISS nearest-neighbour search |
| 🌐 Knowledge Graph | NetworkX graph of papers ↔ keywords ↔ relationships |
| 💬 RAG Q&A | Natural language questions answered with source citations |
| 📈 Visualise | Plotly heatmaps, bar charts, interactive graph (PyVis/Plotly) |

---

## Quick Start

### Local (Python)

```bash
# 1. Clone / unzip
cd ResearchPaperExplorer

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set your Anthropic API key for Q&A
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Launch dashboard
streamlit run src/dashboard.py
```

Open **http://localhost:8501** in your browser.

---

### Docker

```bash
# Build and run
docker-compose up --build

# With API key
ANTHROPIC_API_KEY=sk-ant-... docker-compose up --build
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```
---

## Module Reference

### `PDFProcessor`
```python
from src.pdf_processor import PDFProcessor

proc = PDFProcessor(chunk_size=500, chunk_overlap=50)
doc = proc.process_file("paper.pdf")       # local file
doc = proc.process_url("https://arxiv.org/abs/2301.00001")
doc = proc.process_bytes(pdf_bytes)        # raw bytes (Streamlit)

doc.cleaned_text   # cleaned full text
doc.chunks         # list[str] of sliding-window chunks
doc.num_pages      # int
```

### `MetadataExtractor`
```python
from src.metadata_extractor import MetadataExtractor

ext = MetadataExtractor()
meta = ext.extract(doc.cleaned_text, source="paper.pdf")

meta.title     # str
meta.authors   # list[str]
meta.abstract  # str
meta.year      # int | None
meta.keywords  # list[str]
meta.to_dict() # dict
```

### `KeywordExtractor`
```python
from src.keyword_extractor import KeywordExtractor

kw = KeywordExtractor(top_n=15)
keywords = kw.extract(text)               # single text
keywords = kw.extract_from_chunks(chunks) # multi-chunk
```

### `EmbeddingGenerator`
```python
from src.embedding_generator import EmbeddingGenerator

gen = EmbeddingGenerator("all-MiniLM-L6-v2")
vec = gen.embed_text("query text")        # shape (384,)
mat = gen.embed_batch(["a", "b", "c"])   # shape (3, 384)
doc_vec = gen.embed_chunks(chunks)        # mean-pooled (384,)
```

### `SimilarityEngine`
```python
from src.similarity_engine import SimilarityEngine

eng = SimilarityEngine()
eng.add_paper("p1", embedding, title="Attention Is All You Need")
eng.add_paper("p2", embedding2, title="BERT")

score = eng.paper_similarity("p1", "p2")          # float
similar = eng.most_similar("p1", top_k=5)          # list[SimilarityResult]
hits = eng.search_by_embedding(query_vec, top_k=5) # semantic search
matrix = eng.similarity_matrix()                   # np.ndarray N×N
```

### `KnowledgeGraphBuilder`
```python
from src.graph_builder import KnowledgeGraphBuilder

gb = KnowledgeGraphBuilder()
gb.add_paper("p1", "Title", authors=["A"], year=2023, keywords=["ml"])
gb.add_similarity_edge("p1", "p2", score=0.87)

gb.papers_with_keyword("ml")      # list[str]
gb.similar_papers("p1")           # list[(id, score)]
gb.to_pyvis_html()                # interactive HTML
gb.to_plotly_data()               # dict for Plotly scatter
```

### `RAGEngine`
```python
from src.rag_engine import RAGEngine

rag = RAGEngine(embedding_generator=emb_gen, api_key="sk-ant-...")
rag.add_chunks(chunks, paper_id="p1", paper_title="My Paper")

answer = rag.ask("Which papers discuss attention mechanisms?")
print(answer.answer)
for src in answer.sources:
    print(src.paper_title, src.score)
```

---

## Supported Domains

- Artificial Intelligence / Machine Learning
- Physics / Astrophysics
- Data Science / Statistics
- Any English-language academic PDF

---

## Configuration

| Env Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for LLM-powered Q&A (Claude claude-sonnet-4-20250514) |

---

