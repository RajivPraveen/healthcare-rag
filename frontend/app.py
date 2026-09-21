"""Module 12 + 10 - Streamlit frontend.

Three tabs:

* **Ask**        question -> grounded answer, expandable citations, live timings
* **Corpus**     what is actually indexed, and why an answer could exist at all
* **Evaluation** the Module 10 dashboard, rendered from saved experiment JSON

Talks to the FastAPI backend over HTTP when it is reachable, and otherwise
falls back to calling the pipeline in-process. That means the UI is
demonstrable with one command, but still exercises the real API path when the
full stack is running under docker-compose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run frontend/app.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import get_settings  # noqa: E402

# Prefer the process env, then .env via Settings, then the compose default.
# Port 8000 is a popular default and another local service answering there
# would be mistaken for this API — api_available() also checks the payload.
API_BASE = os.getenv("API_BASE_URL") or get_settings().api_base_url

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOURCE_TYPE_LABELS = {
    "research_paper": "📄 Research paper",
    "drug_label": "💊 FDA drug label",
    "guideline": "📋 Guideline",
    "standard": "📐 Standard",
    "other": "📁 Other",
}


# ---------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def local_pipeline():
    """In-process fallback when the API is not running."""
    from src.generation.pipeline import RAGConfig, RAGPipeline
    from src.retrieval.index import RagIndex

    index = RagIndex.load("default")
    pipeline = RAGPipeline(index, config=RAGConfig())
    pipeline.warmup()
    return pipeline


def api_available() -> bool:
    """Check that *this* project's API is at API_BASE.

    A bare 200 on /health is not enough: port 8000 is a popular default and
    another service answering there would be mistaken for the backend, then
    fail with a confusing 404 on the first query. Look for a field only this
    API returns.
    """
    try:
        response = httpx.get(f"{API_BASE}/health", timeout=2.0)
        return response.status_code == 200 and "index_loaded" in response.json()
    except Exception:
        return False


def run_query(question: str, strategy: str, rerank: bool, top_k: int, top_n: int) -> dict[str, Any]:
    payload = {
        "question": question,
        "strategy": strategy,
        "rerank": rerank,
        "top_k": top_k,
        "top_n": top_n,
    }
    if st.session_state.get("use_api"):
        response = httpx.post(f"{API_BASE}/query", json=payload, timeout=180.0)
        response.raise_for_status()
        return response.json()

    from src.generation.pipeline import RAGConfig, RAGPipeline

    base = local_pipeline()
    config = RAGConfig(strategy=strategy, rerank=rerank, top_k=top_k, top_n=top_n)
    pipeline = RAGPipeline(base.index, config=config, llm=base.llm, embedder=base.embedder)
    result = pipeline.answer(question)
    return {
        "answer": result.answer,
        "sources": [c.model_dump() for c in result.citations],
        "confidence": result.confidence,
        "grounded": result.grounded,
        "strategy": result.strategy,
        "model": result.model,
        "retrieval_latency": result.timings.retrieval_latency,
        "rerank_latency": result.timings.rerank_latency,
        "generation_latency": result.timings.generation_latency,
        "total_latency": result.timings.total_latency,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "estimated_cost_usd": result.usage.estimated_cost_usd,
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_documents() -> list[dict[str, Any]]:
    if st.session_state.get("use_api"):
        try:
            return httpx.get(f"{API_BASE}/documents", timeout=30.0).json()["documents"]
        except Exception:
            pass
    try:
        return local_pipeline().index.documents()
    except Exception:
        return []


@st.cache_data(ttl=30, show_spinner=False)
def load_evaluation() -> dict[str, Any]:
    import json

    results_dir = get_settings().eval_dir / "results"
    payload: dict[str, Any] = {}
    for key, name in (
        ("strategy_comparison", "strategy_comparison.json"),
        ("chunk_sweep", "chunk_sweep.json"),
    ):
        path = results_dir / name
        payload[key] = json.loads(path.read_text()) if path.exists() else None
    return payload


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if "use_api" not in st.session_state:
    st.session_state.use_api = api_available()

with st.sidebar:
    st.title("🩺 Healthcare RAG")
    st.caption("Grounded clinical Q&A with citations")

    st.subheader("Retrieval")
    strategy = st.selectbox(
        "Strategy",
        ["hybrid", "vector", "bm25", "hybrid_weighted"],
        format_func=lambda s: {
            "hybrid": "Hybrid (RRF)",
            "vector": "Vector only",
            "bm25": "BM25 only",
            "hybrid_weighted": "Hybrid (weighted)",
        }[s],
        help="Hybrid fuses dense and lexical results by reciprocal rank.",
    )
    rerank = st.toggle(
        "Cross-encoder reranking",
        value=True,
        help="Rescores the candidate pool with a query-passage cross-encoder.",
    )
    top_k = st.slider("Candidates retrieved (K)", 5, 50, 20, 5)
    top_n = st.slider("Passages sent to the LLM (N)", 1, 10, 5)

    st.divider()
    st.subheader("Backend")
    st.session_state.use_api = st.toggle(
        "Use FastAPI backend", value=st.session_state.use_api, help=API_BASE
    )
    if not st.session_state.use_api:
        st.caption("🟡 Running the pipeline in-process")
    elif api_available():
        # Probe rather than trust the toggle: claiming the API is reachable
        # because the switch is on would hide the actual failure.
        st.caption(f"🟢 API reachable at {API_BASE}")
    else:
        st.caption(f"🔴 No API at {API_BASE} — start it with `hcrag serve`")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

ask_tab, corpus_tab, eval_tab = st.tabs(["💬 Ask", "📚 Corpus", "📊 Evaluation"])


with ask_tab:
    st.header("Ask a clinical question")

    examples = [
        "What are the recommended first-line treatments for hypertension?",
        "What are the most common adverse reactions to amlodipine?",
        "Which anticoagulants are recommended for stroke prevention in atrial fibrillation?",
        "What is the mechanism of action of SGLT2 inhibitors?",
        "What are the major risk factors for chronic kidney disease?",
    ]
    # Drive the textarea through session state rather than the `value=`
    # argument. With `value=` the widget is reset from that argument on every
    # rerun, so typed input could be lost and picking an example after typing
    # would silently clobber it.
    st.session_state.setdefault("question_text", "")

    picked = st.pills("Examples", examples, selection_mode="single", default=None)
    if picked and picked != st.session_state.get("last_example"):
        st.session_state.last_example = picked
        st.session_state.question_text = picked

    question = st.text_area(
        "Your question",
        key="question_text",
        placeholder="e.g. What are the first-line treatments for type 2 diabetes?",
        height=90,
        label_visibility="collapsed",
    )

    if st.button("Ask question", type="primary", use_container_width=True):
        if not (question or "").strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving evidence and generating a grounded answer…"):
                    result = run_query(question, strategy, rerank, top_k, top_n)
                st.session_state.result = result
            except Exception as exc:
                st.error(f"Query failed: {exc}")

    result = st.session_state.get("result")
    if result:
        grounded = result.get("grounded", False)
        confidence = result.get("confidence", 0.0)

        if not grounded:
            st.warning(
                "**Not grounded.** The system did not find supporting evidence and declined "
                "to answer rather than guessing."
            )

        st.markdown("### Answer")
        st.markdown(result["answer"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", f"{confidence:.0%}")
        c2.metric("Sources cited", len(result.get("sources", [])))
        c3.metric("Total latency", f"{result['total_latency']:.2f} s")
        c4.metric("Tokens", result.get("prompt_tokens", 0) + result.get("completion_tokens", 0))

        sources = result.get("sources", [])
        if sources:
            st.markdown("### Sources")
            for source in sources:
                label = (
                    f"**[{source['marker']}]** {source['title']} — "
                    f"page {source['page']} · {source['section']}"
                )
                with st.expander(label):
                    st.markdown(f"> {source['snippet']}")
                    meta = f"`{source['document']}`"
                    if source.get("score") is not None:
                        meta += f" · relevance {source['score']:.3f}"
                    st.caption(meta)
                    if source.get("url"):
                        st.link_button("Open original source", source["url"])

        with st.expander("⏱️ Performance breakdown"):
            timings = pd.DataFrame(
                {
                    "Stage": ["Retrieval", "Reranking", "Generation"],
                    "Seconds": [
                        result["retrieval_latency"],
                        result.get("rerank_latency", 0.0),
                        result["generation_latency"],
                    ],
                }
            )
            fig = go.Figure(
                go.Bar(
                    x=timings["Seconds"],
                    y=timings["Stage"],
                    orientation="h",
                    marker_color=["#4C9BE8", "#8E7CC3", "#5BB98C"],
                    text=[f"{v:.3f}s" for v in timings["Seconds"]],
                    textposition="auto",
                )
            )
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="seconds")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Strategy `{result['strategy']}` · model `{result['model']}` · "
                f"estimated cost ${result.get('estimated_cost_usd', 0):.6f}"
            )


with corpus_tab:
    st.header("Indexed corpus")
    documents = load_documents()

    if not documents:
        st.info("No documents indexed yet. Run `hcrag ingest` then `hcrag index`.")
    else:
        frame = pd.DataFrame(documents)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", len(frame))
        c2.metric("Chunks", int(frame["n_chunks"].sum()))
        c3.metric("Pages", int(frame["pages"].sum()))
        c4.metric("Source types", frame["source_type"].nunique())

        counts = frame["source_type"].value_counts()
        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=[SOURCE_TYPE_LABELS.get(s, s) for s in counts.index],
                orientation="h",
                marker_color="#4C9BE8",
                text=counts.values,
                textposition="auto",
            )
        )
        fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        search = st.text_input("Filter by title", placeholder="hypertension…")
        view = frame
        if search:
            view = view[view["title"].str.contains(search, case=False, na=False)]

        st.dataframe(
            view[["title", "source_type", "n_chunks", "pages", "n_sections"]].rename(
                columns={
                    "title": "Title",
                    "source_type": "Type",
                    "n_chunks": "Chunks",
                    "pages": "Pages",
                    "n_sections": "Sections",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=420,
        )


with eval_tab:
    st.header("Evaluation dashboard")
    st.caption(
        "Generated by `hcrag compare` and `hcrag sweep`. "
        "Retrieval metrics are document-level against a labelled test set."
    )
    data = load_evaluation()

    comparison = data.get("strategy_comparison")
    if not comparison:
        st.info("No comparison yet. Run `hcrag compare` to populate this dashboard.")
    else:
        arms = comparison["arms"]
        slices = comparison.get("slices") or {}
        sizes = comparison.get("slice_sizes") or {}

        # The questions are LLM-generated from the passages they are meant to
        # find, so many of them reuse the passage's vocabulary. Those reward
        # string matching rather than retrieval, so the low-leakage slice is
        # the honest default and the full set is kept only for contrast.
        SLICES = {
            "Low leakage (recommended)": "low_leakage",
            "All questions": "all",
            "High leakage": "high_leakage",
        }
        available = {k: v for k, v in SLICES.items() if v in slices}
        if available:
            label = st.radio(
                "Question slice",
                list(available),
                horizontal=True,
                help=(
                    "Questions are generated from the passage they should retrieve. "
                    "Where the question reuses the passage's wording, keyword search "
                    "gets the answer handed to it. The low-leakage slice drops those."
                ),
            )
            slice_key = available[label]
            retrieval_by_arm = slices[slice_key]
            n_slice = sizes.get(slice_key)
        else:
            slice_key = "all"
            retrieval_by_arm = {n: r["retrieval_document"] for n, r in arms.items()}
            n_slice = None

        rows = []
        for name, report in arms.items():
            retrieval = retrieval_by_arm.get(name, {})
            generation = report["generation"]
            engineering = report["engineering"]
            rows.append(
                {
                    "Strategy": name,
                    "Recall@1": retrieval.get("recall@1", 0),
                    "Recall@5": retrieval.get("recall@5", 0),
                    "MRR": retrieval.get("mrr", 0),
                    "nDCG@5": retrieval.get("ndcg@5", 0),
                    "Faithfulness": generation.get("faithfulness") or 0,
                    "Answer relevance": generation.get("answer_relevance") or 0,
                    "Citation accuracy": generation.get("citation_accuracy") or 0,
                    "Latency (s)": engineering.get("mean_total_latency", 0),
                }
            )
        frame = pd.DataFrame(rows)

        # Recall@5 saturates at 1.000 for every arm on the honest slice, so it
        # cannot rank them. MRR still separates "first hit" from "fifth hit".
        best = frame.loc[frame["MRR"].idxmax()]
        counted = f" on {n_slice} questions" if n_slice else ""
        st.success(
            f"**Best retrieval{counted}:** {best['Strategy']} — "
            f"MRR {best['MRR']:.3f}, Recall@1 {best['Recall@1']:.1%}, "
            f"Recall@5 {best['Recall@5']:.1%}"
        )
        if slice_key == "high_leakage":
            st.warning(
                "These questions echo the wording of the passage they should retrieve. "
                "BM25 scores highest here by matching strings, not by understanding the "
                "question — treat this slice as a control, not as a result."
            )

        metric = st.selectbox(
            "Metric",
            ["MRR", "Recall@1", "Recall@5", "nDCG@5", "Faithfulness", "Answer relevance"],
        )
        fig = go.Figure(
            go.Bar(
                x=frame["Strategy"],
                y=frame[metric],
                marker_color="#4C9BE8",
                text=[f"{v:.3f}" for v in frame[metric]],
                textposition="auto",
            )
        )
        fig.update_layout(
            height=340, yaxis_title=metric, margin=dict(l=0, r=0, t=20, b=0), yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            frame.style.format(
                {c: "{:.3f}" for c in frame.columns if c != "Strategy"}
            ).background_gradient(cmap="Greens", subset=["Recall@1", "MRR", "Faithfulness"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Retrieval metrics follow the selected slice. Generation metrics and latency "
            "are measured once per arm on a subsample, so they do not vary by slice; "
            "arms that were not generated on show 0."
        )

    sweep = data.get("chunk_sweep")
    if sweep:
        st.subheader("Chunk size experiment")
        sweep_frame = pd.DataFrame(sweep["rows"])
        fig = go.Figure()
        for metric in ("recall@5", "recall@20", "mrr"):
            fig.add_trace(
                go.Scatter(
                    x=sweep_frame["target_tokens"],
                    y=sweep_frame[metric],
                    mode="lines+markers",
                    name=metric,
                )
            )
        fig.update_layout(
            height=340,
            xaxis_title="chunk size (tokens)",
            yaxis_title="score",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            sweep_frame.style.format(
                {
                    "mean_tokens": "{:.0f}",
                    "index_seconds": "{:.1f}",
                    "query_ms": "{:.1f}",
                    **{
                        c: "{:.3f}"
                        for c in sweep_frame.columns
                        if c.startswith(("recall", "precision", "mrr", "ndcg"))
                    },
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
