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
    page_title="Healthcare RAG · answers from real sources",
    page_icon=":material/medical_information:",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOURCE_TYPE_LABELS = {
    "research_paper": "Research paper",
    "drug_label": "FDA drug label",
    "guideline": "Guideline",
    "standard": "Standard",
    "other": "Other",
}

# ---------------------------------------------------------------------------
# Look and feel: calm and clinical. Neutral surfaces, one blue-teal accent for
# the UI, and a muted data palette validated for colour-blind safety (the best
# result is highlighted in slot 1; the rest recede to grey).
# ---------------------------------------------------------------------------

ACCENT = "#0b6e8a"
DATA = "#3b6ea8"
SERIES = ["#3b6ea8", "#d0643c", "#2f9a7e"]
MUTED_BAR = "#c9ced3"
INK, INK_2, GRID, AXIS = "#18212a", "#4a5561", "#e9ecef", "#d3d8dd"
FONT = 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif'

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea {{ font-family: {FONT}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ letter-spacing: -0.015em; }}
.hc-head .kicker {{ color: {ACCENT}; text-transform: uppercase; letter-spacing: .12em; font-size: 11.5px; font-weight: 600; }}
.hc-head h1 {{ font-size: 30px; line-height: 1.2; font-weight: 650; margin: 4px 0 6px 0; padding: 0; }}
.hc-head p {{ font-size: 15px; margin: 0 0 16px 0; max-width: 860px; line-height: 1.55; opacity: .85; }}
.hc-qa {{ display: grid; grid-template-columns: 1fr 1.3fr; margin: 0 0 18px 0; border: 1px solid rgba(128,128,128,.25);
          border-radius: 12px; overflow: hidden; }}
.hc-qa > div {{ padding: 14px 18px; }}
.hc-qa > div + div {{ border-left: 1px solid rgba(128,128,128,.25); }}
.hc-qa .lab {{ opacity: .6; text-transform: uppercase; letter-spacing: .08em; font-size: 11px; font-weight: 600; }}
.hc-qa .txt {{ font-size: 15px; margin-top: 4px; line-height: 1.5; }}
@media (max-width: 800px) {{ .hc-qa {{ grid-template-columns: 1fr; }} .hc-qa > div + div {{ border-left: 0; border-top: 1px solid rgba(128,128,128,.25); }} }}
.hc-read {{ font-size: 13.5px; margin: -4px 0 14px 0; line-height: 1.5; opacity: .85; }}
.hc-note {{ border-radius: 10px; padding: 11px 15px; margin: 6px 0 14px 0; border: 1px solid rgba(128,128,128,.25);
            border-left: 3px solid {ACCENT}; font-size: 14px; line-height: 1.55; }}
div[data-testid="stMetric"] {{ border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 10px 14px; }}
div[data-testid="stExpander"] details {{ border-radius: 10px; }}
</style>
""",
    unsafe_allow_html=True,
)


def header(title: str, subtitle: str, kicker: str, question: str = "", answer: str = "") -> None:
    st.markdown(
        f'<div class="hc-head"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )
    if question:
        st.markdown(
            f'<div class="hc-qa"><div><div class="lab">The question</div><div class="txt">{question}</div></div>'
            f'<div><div class="lab">The short answer</div><div class="txt">{answer}</div></div></div>',
            unsafe_allow_html=True,
        )


def how_to_read(text_html: str) -> None:
    st.markdown(f'<div class="hc-read"><b>How to read this:</b> {text_html}</div>', unsafe_allow_html=True)


def note(text_html: str) -> None:
    st.markdown(f'<div class="hc-note">{text_html}</div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 320, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=20, b=0), font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **layout,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS)
    return fig


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

STRATEGY_NAMES = {
    "hybrid": "Hybrid: meaning + keywords (recommended)",
    "vector": "Meaning only (vector search)",
    "bm25": "Keywords only (BM25)",
    "hybrid_weighted": "Hybrid, weighted scores",
}

with st.sidebar:
    st.markdown("**Healthcare RAG**")
    st.caption(
        "An AI that answers medical questions using only a library of real research papers and FDA drug "
        "labels, and shows the source for every claim."
    )
    st.caption("Not medical advice: a tool for exploring the literature.")
    st.divider()
    with st.expander("Advanced settings", icon=":material/tune:"):
        strategy = st.selectbox(
            "How to search the library",
            ["hybrid", "vector", "bm25", "hybrid_weighted"],
            format_func=STRATEGY_NAMES.get,
            help="Hybrid combines meaning-based (vector) and keyword (BM25) search by merging their rankings.",
        )
        rerank = st.toggle(
            "Re-check the top results (reranking)",
            value=True,
            help="A second model re-reads each candidate passage next to the question and re-orders them.",
        )
        top_k = st.slider("Passages to consider", 5, 50, 20, 5, help="Candidates retrieved before re-checking (K).")
        top_n = st.slider("Passages given to the AI", 1, 10, 5, help="How many passages the answer is written from (N).")
        st.divider()
        st.session_state.use_api = st.toggle(
            "Use the API server", value=st.session_state.use_api, help=API_BASE
        )
        if not st.session_state.use_api:
            st.caption("Running inside this app (no API server needed)")
        elif api_available():
            # Probe rather than trust the toggle: claiming the API is reachable
            # because the switch is on would hide the actual failure.
            st.caption(f"Connected to the API at {API_BASE}")
        else:
            st.caption(f"No API at {API_BASE}. Start it with `hcrag serve`.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

ask_tab, corpus_tab, eval_tab = st.tabs(["Ask a question", "What's in the library", "How accurate is it?"])


with ask_tab:
    header(
        "Ask a medical question",
        "Every answer is written <b>only</b> from a library of 289 real research papers and FDA drug labels, "
        "with a numbered source for each claim. If the library doesn't contain the answer, it is built to say so "
        "instead of guessing.",
        "Healthcare RAG",
    )

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

    picked = st.pills("Try an example", examples, selection_mode="single", default=None)
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

    if st.button("Ask", type="primary", use_container_width=True, icon=":material/search:"):
        if not (question or "").strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching the library and writing an answer from what it finds…"):
                    result = run_query(question, strategy, rerank, top_k, top_n)
                st.session_state.result = result
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

    result = st.session_state.get("result")
    if result:
        grounded = result.get("grounded", False)
        confidence = result.get("confidence", 0.0)
        sources = result.get("sources", [])

        if not grounded:
            note(
                "<b>Not in the library.</b> No passage clearly supports an answer, so the system declined "
                "rather than guessing. Try rephrasing, or ask about a condition or drug the library covers "
                "(see <i>What's in the library</i>)."
            )

        st.markdown("### Answer")
        st.markdown(result["answer"])
        if grounded and sources:
            how_to_read(
                "each <b>[number]</b> points to a source below. Open a source to read the exact passage the "
                "claim came from."
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("How well the sources match", f"{confidence:.0%}",
                  help="The system's confidence that the retrieved passages answer the question.")
        c2.metric("Sources used", len(sources))
        c3.metric("Time to answer", f"{result['total_latency']:.1f} s")

        if sources:
            st.markdown("### Sources")
            doc_types = {d.get("document_id"): d.get("source_type") for d in load_documents()}
            for source in sources:
                kind = SOURCE_TYPE_LABELS.get(doc_types.get(source.get("document_id"), ""), "")
                label = f"[{source['marker']}]  {source['title']}  ·  page {source['page']}, {source['section']}"
                with st.expander(label):
                    st.markdown(f"> {source['snippet']}")
                    meta = " · ".join(x for x in (kind, f"`{source['document']}`") if x)
                    if source.get("score") is not None:
                        meta += f" · match score {source['score']:.2f}"
                    st.caption(meta)
                    if source.get("url"):
                        st.link_button("Open the original", source["url"], icon=":material/open_in_new:")

        with st.expander("Technical details: timing, model and cost"):
            timings = pd.DataFrame(
                {
                    "Stage": ["Searching the library", "Re-checking results", "Writing the answer"],
                    "Seconds": [
                        result["retrieval_latency"],
                        result.get("rerank_latency", 0.0),
                        result["generation_latency"],
                    ],
                }
            )
            fig = go.Figure(
                go.Bar(
                    x=timings["Seconds"], y=timings["Stage"], orientation="h", marker_color=DATA,
                    text=[f"{v:.2f} s" for v in timings["Seconds"]], textposition="auto",
                    hovertemplate="%{y}: <b>%{x:.3f} s</b><extra></extra>",
                )
            )
            st.plotly_chart(style(fig, 200, xaxis_title="seconds", yaxis_autorange="reversed"), use_container_width=True)
            tokens = result.get("prompt_tokens", 0) + result.get("completion_tokens", 0)
            st.caption(
                f"Search: {STRATEGY_NAMES.get(result['strategy'], result['strategy'])} · model `{result['model']}` · "
                f"{tokens:,} tokens · estimated cost ${result.get('estimated_cost_usd', 0):.4f}"
            )

    st.caption(
        "Decision support for exploring the literature, not a medical device or patient advice. An AI can still "
        "misread a passage, so check every claim against its source."
    )


with corpus_tab:
    documents = load_documents()
    if not documents:
        header("What's in the library", "The answers can only come from these documents.", "The library")
        st.info("No documents indexed yet. Run `hcrag ingest` then `hcrag index`.")
    else:
        frame = pd.DataFrame(documents)
        counts = frame["source_type"].value_counts()
        papers, labels = int(counts.get("research_paper", 0)), int(counts.get("drug_label", 0))
        header(
            "What's in the library",
            "The answers can only come from these documents: open-access medical research papers from PubMed "
            "Central and official FDA drug labels, split into short passages so each claim can point to its "
            "exact page and section.",
            "The library",
            question="What can this system actually answer?",
            answer=f"Questions covered by <b>{len(frame):,} documents</b>: <b>{papers}</b> research papers and "
                   f"<b>{labels}</b> FDA drug labels, split into <b>{int(frame['n_chunks'].sum()):,}</b> searchable "
                   "passages. For anything outside them, it is built to reply “not in the library” instead of guessing.",
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", f"{len(frame):,}")
        c2.metric("Research papers", papers)
        c3.metric("FDA drug labels", labels)
        c4.metric("Searchable passages", f"{int(frame['n_chunks'].sum()):,}")

        search = st.text_input("Search titles", placeholder="e.g. hypertension, diabetes, amlodipine")
        view = frame
        if search:
            view = view[view["title"].str.contains(search, case=False, na=False)]
        st.caption(f"Showing {len(view):,} of {len(frame):,} documents")
        st.dataframe(
            view.assign(source_type=view["source_type"].map(lambda s: SOURCE_TYPE_LABELS.get(s, s)))[
                ["title", "source_type", "pages", "n_chunks"]
            ].rename(columns={"title": "Title", "source_type": "Type", "pages": "Pages", "n_chunks": "Passages"}),
            use_container_width=True,
            hide_index=True,
            height=460,
        )


with eval_tab:
    data = load_evaluation()
    comparison = data.get("strategy_comparison")
    if not comparison:
        header("How accurate is it?", "No test results yet.", "Accuracy")
        st.info("No comparison yet. Run `hcrag compare` to populate this dashboard.")
    else:
        arms = comparison["arms"]
        slices = comparison.get("slices") or {}
        sizes = comparison.get("slice_sizes") or {}
        best_arm = "Hybrid + Reranker" if "Hybrid + Reranker" in arms else list(arms)[-1]
        gen = arms[best_arm]["generation"]
        eng = arms[best_arm]["engineering"]
        low = (slices.get("low_leakage") or {}).get(best_arm, {})
        header(
            "How accurate is it?",
            "The system was tested on 76 questions with known answers. Two things are measured: does it "
            "<b>find</b> the right source, and does the <b>answer</b> stick to what the sources say?",
            "Accuracy",
            question="Can the answers be trusted?",
            answer=(f"It ranks the right source first <b>{low.get('recall@1', 0):.0%}</b> of the time on "
                    f"{sizes.get('low_leakage', 'the')} fair test questions, and <b>{gen.get('faithfulness', 0):.0%}</b> of what it writes is supported by its "
                    f"sources. Every citation it gave pointed to a passage it really found "
                    f"(<b>{gen.get('citation_accuracy', 0):.0%}</b>). A typical answer takes "
                    f"<b>{eng.get('mean_total_latency', 0):.1f} s</b>.") if low else "See the results below.",
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Right source first", f"{low.get('recall@1', 0):.0%}", help="Recall@1 on the fair test questions")
        c2.metric("Claims backed by sources", f"{gen.get('faithfulness', 0):.0%}", help="Faithfulness, judged by a separate AI model")
        c3.metric("Citations that are real", f"{gen.get('citation_accuracy', 0):.0%}", help="Citation accuracy: each [number] points to a passage it actually retrieved, checked automatically")
        c4.metric("Wrongly refused", f"{gen.get('false_refusal_rate', 0):.0%}", help="False refusal rate: answerable questions it declined")

        st.markdown("### Which search method finds the right source best?")
        # The questions are LLM-generated from the passages they are meant to
        # find, so many of them reuse the passage's vocabulary. Those reward
        # string matching rather than retrieval, so the low-leakage slice is
        # the honest default and the full set is kept only for contrast.
        SLICES = {
            "Fair questions (recommended)": "low_leakage",
            "All questions": "all",
            "Questions that copy the source's wording": "high_leakage",
        }
        available = {k: v for k, v in SLICES.items() if v in slices}
        if available:
            label = st.radio(
                "Test questions",
                list(available),
                horizontal=True,
                help=(
                    "Test questions were written by an AI from the passage they should find. Many reuse that "
                    "passage's exact words, which lets keyword search 'cheat'. The fair set leaves those out."
                ),
            )
            slice_key = available[label]
            retrieval_by_arm = slices[slice_key]
            n_slice = sizes.get(slice_key)
        else:
            slice_key = "all"
            retrieval_by_arm = {n: r["retrieval_document"] for n, r in arms.items()}
            n_slice = None

        METRICS = {
            "How high the right source ranks (MRR)": "mrr",
            "Right source ranked first (Recall@1)": "recall@1",
            "Right source in the top 5 (Recall@5)": "recall@5",
            "Ranking quality of the top 5 (nDCG@5)": "ndcg@5",
        }
        metric_label = st.selectbox("Measure", list(METRICS))
        metric = METRICS[metric_label]
        names = list(arms)
        values = [retrieval_by_arm.get(n, {}).get(metric, 0) for n in names]
        top = max(range(len(values)), key=values.__getitem__)
        fig = go.Figure(
            go.Bar(
                x=names, y=values, marker_color=[DATA if i == top else MUTED_BAR for i in range(len(values))],
                text=[f"{v:.1%}" for v in values], textposition="outside", width=0.5,
                hovertemplate="%{x}: <b>%{y:.1%}</b><extra></extra>",
            )
        )
        st.plotly_chart(style(fig, 330, yaxis=dict(range=[0, 1.12], tickformat=".0%"), showlegend=False),
                        use_container_width=True)
        counted = f" ({n_slice} questions)" if n_slice else ""
        how_to_read(
            f"each bar is one way of searching the library{counted}; the best is highlighted. "
            "<b>Vector</b> searches by meaning, <b>BM25</b> by matching keywords, <b>Hybrid</b> combines both, and "
            "the <b>Reranker</b> re-reads the top candidates to put the best one first."
        )
        if slice_key == "high_leakage":
            note(
                "<b>Treat this set as a control.</b> These questions reuse the wording of the passage they should "
                "find, so keyword search scores highest by matching words, not by understanding the question."
            )
        else:
            note(
                "<b>The test almost fooled us.</b> On all questions, plain keyword search looked best. The reason: "
                "most test questions were written from the answer passage and copied its words. On the fair "
                "questions that don't, keyword search loses its edge and the hybrid + reranker method wins."
            )

        with st.expander("For analysts: every measure, every method"):
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
            st.dataframe(
                frame.style.format({c: "{:.3f}" for c in frame.columns if c != "Strategy"}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Retrieval measures follow the selected question set. Answer-quality measures and latency are "
                "measured once per method on a 25-question subsample, so they don't change with the set; methods "
                "that weren't run for answer quality show 0."
            )

    sweep = data.get("chunk_sweep")
    if sweep:
        st.markdown("### Does the passage size matter?")
        sweep_frame = pd.DataFrame(sweep["rows"])
        fig = go.Figure()
        for (col, name), color in zip(
            (("mrr", "How high the right source ranks (MRR)"), ("recall@5", "Right source in the top 5"),
             ("recall@20", "Right source in the top 20")), SERIES, strict=True,
        ):
            fig.add_trace(
                go.Scatter(
                    x=sweep_frame["target_tokens"], y=sweep_frame[col], mode="lines+markers", name=name,
                    line=dict(color=color, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
                    hovertemplate=f"{name}: <b>%{{y:.1%}}</b><extra></extra>",
                )
            )
        st.plotly_chart(
            style(fig, 320, hovermode="x unified", xaxis_title="passage size (target tokens, about ¾ of a word each)",
                  yaxis=dict(tickformat=".0%")),
            use_container_width=True,
        )
        how_to_read(
            "the library was split into passages of four sizes and tested each time. The lines are nearly flat: "
            "passage size moves accuracy by only about two points, far less than the choice of search method."
        )
        with st.expander("For analysts: passage-size results"):
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
