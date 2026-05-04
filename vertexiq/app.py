"""Streamlit UI for VertexIQ.

Run with:

    streamlit run vertexiq/app.py

The app keeps a single :class:`KnowledgeGraph` instance in
``st.session_state`` so the graph survives reruns. Everything else
(extraction, querying, visualization) is recomputed on demand.
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# Support both ``streamlit run vertexiq/app.py`` (run as top-level script)
# and ``python -m vertexiq.app`` (run as package member).
try:  # pragma: no cover - import path resolution
    from .extractor import extract_triples
    from .graph_builder import KnowledgeGraph
    from .query_engine import QueryEngine
except ImportError:  # pragma: no cover
    from extractor import extract_triples  # type: ignore[no-redef]
    from graph_builder import KnowledgeGraph  # type: ignore[no-redef]
    from query_engine import QueryEngine  # type: ignore[no-redef]

GRAPH_PATH = os.getenv("VERTEXIQ_GRAPH_PATH", "graph.json")


def _get_graph() -> KnowledgeGraph:
    """Return the session-scoped :class:`KnowledgeGraph` singleton."""
    if "graph" not in st.session_state:
        st.session_state.graph = KnowledgeGraph(path=GRAPH_PATH)
    return st.session_state.graph


def _degree_color(degree: int, max_degree: int) -> str:
    """Map a degree onto a blue-shaded hex color (darker = higher degree)."""
    if max_degree <= 0:
        return "#9ec5fe"
    ratio = min(degree / max_degree, 1.0)
    # interpolate between light blue (#cfe2ff) and deep blue (#0a3d91)
    start = (207, 226, 255)
    end = (10, 61, 145)
    rgb = tuple(int(start[i] + (end[i] - start[i]) * ratio) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _render_graph_html(graph: KnowledgeGraph) -> str:
    """Render the current graph as a self-contained pyvis HTML string."""
    net = Network(
        height="650px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2937",
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut()

    if graph.graph.number_of_nodes() == 0:
        return "<div style='padding:2rem;color:#6b7280'>Graph is empty. Paste text in the sidebar and click <strong>Extract &amp; Build Graph</strong>.</div>"

    degrees = dict(graph.graph.degree())
    max_degree = max(degrees.values()) if degrees else 1

    for node, deg in degrees.items():
        net.add_node(
            node,
            label=node,
            title=f"{node} (degree {deg})",
            color=_degree_color(deg, max_degree),
            value=deg,
        )

    for u, v, data in graph.graph.edges(data=True):
        relation = data.get("relation", "")
        net.add_edge(u, v, label=relation, title=relation, arrows="to")

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        net.write_html(tmp.name, notebook=False, open_browser=False)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _render_sidebar(graph: KnowledgeGraph) -> None:
    """Render the document-ingestion sidebar."""
    st.sidebar.header("Ingest")
    text = st.sidebar.text_area(
        "Paste document text",
        height=240,
        placeholder="Paste a paragraph about the AI industry, regulations, products...",
    )
    if st.sidebar.button("Extract & Build Graph", use_container_width=True):
        if not text.strip():
            st.sidebar.warning("Please paste some text first.")
        else:
            with st.spinner("Extracting triples and updating the graph..."):
                try:
                    triples = extract_triples(text)
                    added = graph.add_triples(triples)
                    graph.save()
                    st.sidebar.success(f"Added {added} new edges (extracted {len(triples)} triples).")
                except RuntimeError as exc:
                    st.sidebar.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"Extraction failed: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.header("Graph stats")
    stats = graph.get_stats()
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Nodes", stats["nodes"])
    col2.metric("Edges", stats["edges"])
    if stats["top_entities"]:
        st.sidebar.markdown("**Top entities (by degree)**")
        for item in stats["top_entities"]:
            st.sidebar.markdown(f"- `{item['entity']}` ({item['degree']})")


def _render_graph_tab(graph: KnowledgeGraph) -> None:
    """Tab 1: interactive pyvis visualization."""
    html = _render_graph_html(graph)
    components.html(html, height=680, scrolling=True)


def _render_query_tab(graph: KnowledgeGraph) -> None:
    """Tab 2: Q&A over the graph."""
    question = st.text_input("Ask a question about the graph", "")
    anchor = st.text_input("Anchor entity (optional)", "")
    run = st.button("Run Query", use_container_width=False)

    if not run:
        return
    if not question.strip():
        st.warning("Please enter a question.")
        return

    engine = QueryEngine(graph)
    with st.spinner("Reasoning over the graph..."):
        try:
            if anchor.strip():
                answer = engine.semantic_query(question, anchor.strip())
            else:
                answer = engine.auto_query(question)
        except ValueError as exc:
            st.error(str(exc))
            return
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Query failed: {exc}")
            return

    st.markdown("### Answer")
    st.write(answer)


def main() -> None:
    """Streamlit entry-point."""
    st.set_page_config(
        page_title="VertexIQ",
        page_icon=":bulb:",
        layout="wide",
    )
    st.title("VertexIQ — Domain Knowledge Graph Explorer")
    st.caption(
        "Extract entity-relationship triples from text, build a persistent "
        "knowledge graph, and ask multi-hop questions grounded in graph "
        "structure."
    )

    graph = _get_graph()
    _render_sidebar(graph)

    tab_graph, tab_query = st.tabs(["Graph View", "Query"])
    with tab_graph:
        _render_graph_tab(graph)
    with tab_query:
        _render_query_tab(graph)


main()
