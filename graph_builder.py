"""
KnowledgeGraphBuilder: Constructs a NetworkX graph of papers, keywords,
and their relationships for interactive exploration.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logger.info("PyVis not available; HTML graph export disabled.")


class KnowledgeGraphBuilder:
    """
    Builds and queries a knowledge graph with three node types:
    - Paper nodes  (type="paper")
    - Keyword nodes (type="keyword")

    Edge types:
    - paper → keyword : "contains"
    - paper → paper   : "similar_to" (weighted by cosine score)
    - keyword → keyword: "co_occurs"
    """

    SIMILARITY_THRESHOLD = 0.60  # Minimum score to draw a paper–paper edge

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------ #
    #  Node & edge construction                                            #
    # ------------------------------------------------------------------ #

    def add_paper(
        self,
        paper_id: str,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None,
        abstract: str = "",
        keywords: list[str] | None = None,
    ) -> None:
        """Add or update a paper node."""
        self.graph.add_node(
            paper_id,
            label=self._short_title(title),
            title_full=title,
            node_type="paper",
            authors=", ".join(authors or []),
            year=year or "",
            abstract=abstract[:300],
        )
        # Add keyword nodes and edges
        for kw in (keywords or []):
            self._add_keyword_node(kw)
            self.graph.add_edge(paper_id, kw, edge_type="contains", weight=1.0)

        # Update keyword co-occurrence edges
        kws = keywords or []
        for i, kw1 in enumerate(kws):
            for kw2 in kws[i + 1:]:
                if self.graph.has_edge(kw1, kw2):
                    self.graph[kw1][kw2]["weight"] = self.graph[kw1][kw2].get("weight", 1) + 1
                else:
                    self.graph.add_edge(kw1, kw2, edge_type="co_occurs", weight=1)

    def add_similarity_edge(
        self,
        id_a: str,
        id_b: str,
        score: float,
    ) -> None:
        """Add a weighted similarity edge between two papers."""
        if score < self.SIMILARITY_THRESHOLD:
            return
        if not (self.graph.has_node(id_a) and self.graph.has_node(id_b)):
            logger.debug("Skipping similarity edge: missing node %s or %s", id_a, id_b)
            return
        self.graph.add_edge(id_a, id_b, edge_type="similar_to", weight=round(score, 4))

    def build_from_collection(
        self,
        papers: list[dict],
        similarity_matrix: Optional["np.ndarray"] = None,
        paper_ids: Optional[list[str]] = None,
    ) -> None:
        """
        Convenience method: build the full graph from a list of paper dicts.

        Each paper dict must have: id, title, authors, year, abstract, keywords.
        Optionally provide a precomputed similarity_matrix and aligned paper_ids.
        """
        for p in papers:
            self.add_paper(
                paper_id=p["id"],
                title=p.get("title", ""),
                authors=p.get("authors", []),
                year=p.get("year"),
                abstract=p.get("abstract", ""),
                keywords=p.get("keywords", []),
            )

        if similarity_matrix is not None and paper_ids:
            import numpy as np
            n = len(paper_ids)
            for i in range(n):
                for j in range(i + 1, n):
                    self.add_similarity_edge(
                        paper_ids[i], paper_ids[j], float(similarity_matrix[i, j])
                    )

    # ------------------------------------------------------------------ #
    #  Query helpers                                                       #
    # ------------------------------------------------------------------ #

    def papers(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "paper"]

    def keywords(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "keyword"]

    def similar_papers(self, paper_id: str) -> list[tuple[str, float]]:
        """Return (neighbour_id, score) for papers similar to paper_id."""
        results = []
        for nbr in self.graph.neighbors(paper_id):
            edge = self.graph[paper_id][nbr]
            if edge.get("edge_type") == "similar_to":
                results.append((nbr, edge["weight"]))
        return sorted(results, key=lambda x: -x[1])

    def papers_with_keyword(self, keyword: str) -> list[str]:
        kw = keyword.lower()
        if not self.graph.has_node(kw):
            return []
        return [
            n for n in self.graph.neighbors(kw)
            if self.graph.nodes[n].get("node_type") == "paper"
        ]

    def get_paper_data(self, paper_id: str) -> dict:
        return dict(self.graph.nodes.get(paper_id, {}))

    def summary(self) -> dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "papers": len(self.papers()),
            "keywords": len(self.keywords()),
            "total_edges": self.graph.number_of_edges(),
            "similarity_edges": sum(
                1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == "similar_to"
            ),
        }

    # ------------------------------------------------------------------ #
    #  Visualisation export                                                #
    # ------------------------------------------------------------------ #

    def to_pyvis_html(
        self,
        height: str = "600px",
        width: str = "100%",
        notebook: bool = False,
    ) -> str:
        """
        Render the graph as a self-contained HTML string using PyVis.
        Returns empty string if PyVis is not installed.
        """
        if not PYVIS_AVAILABLE:
            return ""

        net = Network(height=height, width=width, notebook=notebook, directed=False)
        net.set_options(self._pyvis_options())

        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "keyword")
            label = data.get("label", str(node_id))
            tooltip = self._node_tooltip(node_id, data)

            if node_type == "paper":
                net.add_node(
                    node_id,
                    label=label,
                    title=tooltip,
                    color="#4A90D9",
                    size=25,
                    shape="dot",
                    font={"size": 14, "color": "#ffffff"},
                )
            else:
                net.add_node(
                    node_id,
                    label=label,
                    title=tooltip,
                    color="#F5A623",
                    size=12,
                    shape="diamond",
                    font={"size": 11, "color": "#333333"},
                )

        for src, dst, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "")
            weight = data.get("weight", 1.0)
            if edge_type == "similar_to":
                color = "#4A90D9"
                width = max(1, int(weight * 5))
            elif edge_type == "contains":
                color = "#9B9B9B"
                width = 1
            else:  # co_occurs
                color = "#F5A623"
                width = max(1, min(int(weight), 4))
            net.add_edge(src, dst, color=color, width=width, title=f"{edge_type} ({weight:.2f})")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            html = Path(f.name).read_text()
        return html

    def to_plotly_data(self) -> dict:
        """
        Export graph data suitable for Plotly scatter plots.
        Returns dict with 'nodes' and 'edges' lists.
        """
        pos = nx.spring_layout(self.graph, seed=42, k=0.8)

        nodes = []
        for node_id, (x, y) in pos.items():
            data = self.graph.nodes[node_id]
            nodes.append({
                "id": node_id,
                "x": float(x),
                "y": float(y),
                "label": data.get("label", node_id),
                "type": data.get("node_type", "keyword"),
                "tooltip": self._node_tooltip(node_id, data),
            })

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            if src in pos and dst in pos:
                edges.append({
                    "x0": float(pos[src][0]),
                    "y0": float(pos[src][1]),
                    "x1": float(pos[dst][0]),
                    "y1": float(pos[dst][1]),
                    "type": data.get("edge_type", ""),
                    "weight": data.get("weight", 1.0),
                })

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _add_keyword_node(self, keyword: str) -> None:
        kw = keyword.lower().strip()
        if not self.graph.has_node(kw):
            self.graph.add_node(kw, label=kw, node_type="keyword")

    @staticmethod
    def _short_title(title: str, max_chars: int = 30) -> str:
        return title if len(title) <= max_chars else title[:max_chars].rstrip() + "…"

    @staticmethod
    def _node_tooltip(node_id: str, data: dict) -> str:
        if data.get("node_type") == "paper":
            parts = [f"<b>{data.get('title_full', node_id)}</b>"]
            if data.get("authors"):
                parts.append(f"Authors: {data['authors']}")
            if data.get("year"):
                parts.append(f"Year: {data['year']}")
            if data.get("abstract"):
                parts.append(f"{data['abstract']}…")
            return "<br>".join(parts)
        return f"Keyword: <b>{node_id}</b>"

    @staticmethod
    def _pyvis_options() -> str:
        return json.dumps({
            "physics": {
                "enabled": True,
                "stabilization": {"iterations": 150},
                "barnesHut": {"gravitationalConstant": -8000, "springLength": 200},
            },
            "interaction": {"hover": True, "tooltipDelay": 100},
            "edges": {"smooth": {"type": "dynamic"}},
        })
