"""
VisualizationManager: Produces Plotly figures and HTML graphs for the Streamlit dashboard.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class VisualizationManager:
    """
    Centralises all visualisation logic.
    Methods return Plotly Figure objects or HTML strings.
    """

    # Colour palette
    PAPER_COLOR = "#4A90D9"
    KEYWORD_COLOR = "#F5A623"
    EDGE_COLOR_SIM = "rgba(74, 144, 217, 0.5)"
    EDGE_COLOR_KW = "rgba(245, 166, 35, 0.3)"

    # ------------------------------------------------------------------ #
    #  Similarity heatmap                                                  #
    # ------------------------------------------------------------------ #

    def similarity_heatmap(
        self,
        similarity_matrix,  # np.ndarray N×N
        paper_ids: list[str],
        titles: Optional[list[str]] = None,
    ) -> "go.Figure":
        if not PLOTLY_AVAILABLE:
            raise ImportError("Install plotly: pip install plotly")

        import numpy as np
        labels = titles or paper_ids
        short = [l[:40] + "…" if len(l) > 40 else l for l in labels]

        fig = go.Figure(go.Heatmap(
            z=similarity_matrix,
            x=short,
            y=short,
            colorscale="Blues",
            zmin=0, zmax=1,
            text=np.round(similarity_matrix, 2),
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>vs<br><b>%{x}</b><br>Score: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title="Paper Similarity Matrix",
            xaxis=dict(tickangle=-45, tickfont_size=11),
            yaxis=dict(tickfont_size=11),
            height=max(400, len(labels) * 50),
            margin=dict(l=200, b=200),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#FFFFFF",
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Keyword frequency bar chart                                         #
    # ------------------------------------------------------------------ #

    def keyword_frequency_chart(
        self,
        keyword_counts: dict[str, int],
        top_n: int = 20,
    ) -> "go.Figure":
        if not PLOTLY_AVAILABLE:
            raise ImportError("Install plotly: pip install plotly")

        sorted_kws = sorted(keyword_counts.items(), key=lambda x: -x[1])[:top_n]
        kws, counts = zip(*sorted_kws) if sorted_kws else ([], [])

        fig = go.Figure(go.Bar(
            x=list(counts),
            y=list(kws),
            orientation="h",
            marker_color=self.KEYWORD_COLOR,
            hovertemplate="%{y}: %{x} papers<extra></extra>",
        ))
        fig.update_layout(
            title=f"Top {top_n} Keywords Across Papers",
            xaxis_title="Frequency",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(kws) * 24),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1C1C2E",
            font_color="#FFFFFF",
            margin=dict(l=180),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Similarity scatter / network (Plotly-based)                         #
    # ------------------------------------------------------------------ #

    def graph_plotly(self, graph_data: dict) -> "go.Figure":
        """
        Render the knowledge graph as a Plotly scatter plot.
        graph_data is produced by KnowledgeGraphBuilder.to_plotly_data().
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Install plotly: pip install plotly")

        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        # Edge traces
        edge_traces: list[go.Scatter] = []
        for e in edges:
            color = self.EDGE_COLOR_SIM if e["type"] == "similar_to" else self.EDGE_COLOR_KW
            edge_traces.append(go.Scatter(
                x=[e["x0"], e["x1"], None],
                y=[e["y0"], e["y1"], None],
                mode="lines",
                line=dict(width=max(0.5, e["weight"] * 2), color=color),
                hoverinfo="none",
                showlegend=False,
            ))

        # Node traces — split by type
        paper_nodes = [n for n in nodes if n["type"] == "paper"]
        kw_nodes = [n for n in nodes if n["type"] == "keyword"]

        def make_scatter(node_list, color, size, name):
            return go.Scatter(
                x=[n["x"] for n in node_list],
                y=[n["y"] for n in node_list],
                mode="markers+text",
                marker=dict(size=size, color=color, line=dict(width=1, color="#ffffff")),
                text=[n["label"] for n in node_list],
                textposition="top center",
                textfont=dict(size=10, color="#ffffff"),
                hovertext=[n["tooltip"] for n in node_list],
                hoverinfo="text",
                name=name,
            )

        paper_scatter = make_scatter(paper_nodes, self.PAPER_COLOR, 18, "Papers")
        kw_scatter = make_scatter(kw_nodes, self.KEYWORD_COLOR, 10, "Keywords")

        fig = go.Figure(data=[*edge_traces, paper_scatter, kw_scatter])
        fig.update_layout(
            title="Knowledge Graph",
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#FFFFFF",
            height=600,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Paper metadata table                                                #
    # ------------------------------------------------------------------ #

    def papers_table(self, papers: list[dict]) -> "pd.DataFrame":
        if not PLOTLY_AVAILABLE:
            raise ImportError("Install pandas: pip install pandas")

        rows = []
        for p in papers:
            rows.append({
                "Title": p.get("title", ""),
                "Authors": ", ".join(p.get("authors", [])) or "—",
                "Year": p.get("year", "—"),
                "Keywords": ", ".join(p.get("keywords", [])[:5]),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Similarity bar for a single paper                                   #
    # ------------------------------------------------------------------ #

    def paper_similarity_bar(
        self,
        similarities: list[tuple[str, float]],  # (title, score)
        query_title: str,
    ) -> "go.Figure":
        if not similarities:
            return go.Figure()

        titles, scores = zip(*similarities)
        short_titles = [t[:50] + "…" if len(t) > 50 else t for t in titles]

        fig = go.Figure(go.Bar(
            x=list(scores),
            y=list(short_titles),
            orientation="h",
            marker_color=[
                f"rgba(74,144,217,{max(0.3, s)})" for s in scores
            ],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Papers Similar to: {query_title[:60]}",
            xaxis=dict(title="Cosine Similarity", range=[0, 1]),
            yaxis=dict(autorange="reversed"),
            height=max(250, len(titles) * 30),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1C1C2E",
            font_color="#FFFFFF",
            margin=dict(l=250),
        )
        return fig
