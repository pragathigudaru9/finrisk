"""
FinRisk — Streamlit Dashboard (Single-Company Risk Intelligence)

4-tab dashboard for focused single-company risk analysis:
  1. 📊 Risk Changes     — quarter-over-quarter category spike table + bar chart
  2. 💰 Financial & Sentiment — Altman Z-Score trend + FinBERT sentiment
  3. 📄 Evidence         — exact filing sentences for each flagged risk
  4. 💬 Ask FinRisk      — RAG chatbot backed by Groq Llama-3.3-70b
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinRisk — Single-Company Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e2e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
    border-right: 1px solid #2d3748;
}
section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(30,41,59,0.6);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.5rem !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom: 2px solid #60a5fa; }

/* Expanders */
.streamlit-expanderHeader { color: #f1f5f9 !important; font-weight: 600; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(59,130,246,0.4); }

/* Dataframe */
.dataframe { background: #1e293b !important; }

/* Text input */
.stTextInput > div > div > input {
    background: #1e293b; border: 1px solid #3b82f6; border-radius: 8px; color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: chunk sections from a sections DataFrame ─────────────────────────
def chunk_sections(sections_df: pd.DataFrame) -> list[dict]:
    """Convert a sections DataFrame into a list of chunk dicts for classification."""
    from src.chunk import section_aware_chunker
    chunks = []
    for _, row in sections_df.iterrows():
        meta = {
            "ticker": row["ticker"],
            "year": row.get("year", 0),
            "section_type": row["section_type"],
            "period_label": row.get("period_label", "annual"),
            "filing_type": row.get("filing_type", "10-K"),
        }
        for c in section_aware_chunker(row["raw_text"], meta):
            c["period_label"] = row.get("period_label", "annual")
            chunks.append(c)
    return chunks


def run_boilerplate_filter(chunks: list[dict]) -> list[dict]:
    """Filter out boilerplate chunks using the heuristic is_boilerplate() check."""
    from src.classify import is_boilerplate
    return [c for c in chunks if not is_boilerplate(c.get("text", ""))]


# ── Data loader (cached per ticker) ──────────────────────────────────────────
@st.cache_data(show_spinner="📥 Downloading and processing filings…", ttl=3600)
def load_company_data(ticker: str):
    from src.ingest import download_company_latest
    from src.extract_risks import classify_chunks
    from src.risk_score import get_risk_summary
    from src.sentiment import get_sentiment_summary
    from src.zscore import get_zscore_history

    sections_df = download_company_latest(ticker)
    chunks = chunk_sections(sections_df)
    chunks = run_boilerplate_filter(chunks)

    if not chunks:
        return None, None, None, f"No content extracted for {ticker}. Try another ticker."

    # Zero-shot risk classification (this loads BART — may take ~30s on first run)
    classified = classify_chunks(chunks)
    risk_summary = get_risk_summary(ticker, classified)
    sentiment = get_sentiment_summary(ticker)
    zscore_history = get_zscore_history(ticker)

    return risk_summary, sentiment, zscore_history, None


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    from config import AVAILABLE_TICKERS, COMPANY_NAMES

    with st.sidebar:
        st.markdown("## 📊 FinRisk")
        st.markdown(
            "<small style='color:#94a3b8'>Single-Company Risk Intelligence</small>",
            unsafe_allow_html=True,
        )
        st.divider()

        ticker = st.selectbox(
            "Select Company",
            AVAILABLE_TICKERS,
            index=AVAILABLE_TICKERS.index("AAPL"),
            key="ticker_select",
        )
        company = COMPANY_NAMES.get(ticker, ticker)
        st.caption(f"**{company}** ({ticker})")

        st.divider()
        st.markdown("#### 📐 Research Results")
        st.markdown("""
**Chunking Ablation (Hit@5):**
- Section-Aware: `0.25` · MRR: `0.129`
- Fixed-Size: `0.20` · MRR: `0.096`
- Section-aware improves MRR **+25%**

**RAGAS Evaluation:**
- Faithfulness: `0.977`
- Context Precision: `0.617`
""")
        st.divider()
        st.caption("Data: SEC EDGAR 10-K & 10-Q · NewsAPI · yfinance")
        return ticker


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30,41,59,0.4)",
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(gridcolor="#2d3748", showgrid=True),
    yaxis=dict(gridcolor="#2d3748", showgrid=True),
    margin=dict(l=0, r=0, t=40, b=0),
)


# ── TAB 1: Risk Changes ───────────────────────────────────────────────────────
def render_risk_changes(risk_summary: dict, ticker: str):
    st.subheader("Quarter-over-Quarter Risk Category Changes")
    st.caption(
        "Chunks from the current quarter vs previous quarter 10-Q filings are "
        "classified by risk type. A category is flagged ⚠️ when mentions increase > 50%."
    )

    risk_counts = risk_summary["risk_counts"]
    flagged_set = set(risk_summary["flagged_categories"])

    # Build table rows
    rows = []
    for cat, d in risk_counts.items():
        arrow = "▲" if d["change_pct"] > 0 else ("▼" if d["change_pct"] < 0 else "–")
        status = "⚠️ Elevated" if d["flagged"] else "✓ Stable"
        rows.append({
            "Risk Category": cat,
            "Previous Quarter": d["previous_quarter"],
            "Current Quarter": d["current_quarter"],
            "Change": f"{arrow} {abs(d['change_pct']):.0f}%",
            "Status": status,
        })

    df_table = pd.DataFrame(rows)

    # Highlight flagged rows
    def highlight_row(row):
        if "Elevated" in row["Status"]:
            return ["background-color: rgba(239,68,68,0.15)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_table.style.apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Grouped bar chart
    cats = [r["Risk Category"] for r in rows]
    fig = go.Figure()
    fig.add_bar(
        name="Previous Quarter",
        x=cats,
        y=[r["Previous Quarter"] for r in rows],
        marker_color="#3b82f6",
        opacity=0.85,
    )
    fig.add_bar(
        name="Current Quarter",
        x=cats,
        y=[r["Current Quarter"] for r in rows],
        marker_color="#f87171",
        opacity=0.85,
    )
    fig.update_layout(
        barmode="group",
        title=f"{ticker} — Risk Mentions by Category",
        legend=dict(orientation="h", y=1.1),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    if flagged_set:
        st.warning(
            f"**{len(flagged_set)} category/ies elevated:** {', '.join(sorted(flagged_set))}  "
            f"→ See **Evidence** tab for exact filing sentences."
        )
    else:
        st.success("No risk categories show a significant quarter-over-quarter spike.")


# ── TAB 2: Financial & Sentiment ──────────────────────────────────────────────
def render_financial_sentiment(zscore_history, sentiment, ticker: str):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Financial Health — Altman Z-Score")
        if zscore_history is not None and not zscore_history.empty:
            fig_z = go.Figure()
            fig_z.add_scatter(
                x=zscore_history["period"],
                y=zscore_history["z_score"],
                mode="lines+markers",
                name="Z-Score",
                line=dict(color="#60a5fa", width=2.5),
                marker=dict(size=8, color="#60a5fa"),
            )
            fig_z.add_hline(
                y=2.99, line_dash="dash", line_color="#22c55e",
                annotation_text="Safe Zone (2.99)", annotation_position="top right",
            )
            fig_z.add_hline(
                y=1.81, line_dash="dash", line_color="#ef4444",
                annotation_text="Distress Zone (1.81)", annotation_position="bottom right",
            )
            fig_z.update_layout(
                title=f"{ticker} Altman Z-Score (4 Quarters)",
                yaxis_title="Z-Score",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_z, use_container_width=True)

            # Latest zone badge
            latest_z = zscore_history.iloc[-1]
            z_val = latest_z["z_score"]
            zone = latest_z["zone"]
            zone_color = {"Safe": "🟢", "Grey Zone": "🟡", "Distress": "🔴"}.get(zone, "⚪")
            st.metric("Current Z-Score", f"{z_val:.2f}", f"{zone_color} {zone}")

            # Key financial ratios
            c1, c2, c3 = st.columns(3)
            de = latest_z.get("debt_equity")
            om = latest_z.get("operating_margin")
            rg = latest_z.get("revenue_growth")

            c1.metric("Debt / Equity", f"{de:.2f}" if de is not None else "N/A")
            c2.metric("Operating Margin", f"{om:.1%}" if om is not None else "N/A")
            c3.metric("Revenue Growth", f"{rg:.1%}" if rg is not None else "N/A")
        else:
            st.info("Financial data unavailable for this company.")

    with col2:
        st.subheader("📰 News Sentiment — FinBERT (30 Days)")
        if sentiment is not None and not sentiment.empty:
            fig_s = go.Figure()
            fig_s.add_scatter(
                x=sentiment["date"],
                y=sentiment["rolling_7d"],
                mode="lines",
                name="7-Day Rolling",
                line=dict(color="#a78bfa", width=2),
            )
            fig_s.add_scatter(
                x=sentiment["date"],
                y=sentiment["rolling_30d"],
                mode="lines",
                name="30-Day Rolling",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            )
            fig_s.add_hline(y=0, line_dash="dot", line_color="#4b5563")
            fig_s.update_layout(
                title=f"{ticker} — 7-Day & 30-Day Sentiment",
                yaxis_title="Sentiment Score",
                legend=dict(orientation="h", y=1.1),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_s, use_container_width=True)

            latest_s = sentiment.iloc[-1]
            score_30d = float(latest_s.get("rolling_30d", 0))
            label = (
                "🟢 Bullish" if score_30d > 0.1 else
                "🔴 Bearish" if score_30d < -0.1 else
                "🟡 Neutral"
            )
            st.metric("30-Day Sentiment Score", f"{score_30d:.3f}", label)

            if latest_s.get("sentiment_flag"):
                st.warning(
                    "⚠️ Sentiment deteriorating: 7-day average has dropped below "
                    "the 30-day average by more than 0.15 points."
                )
        else:
            st.info(
                "Live sentiment data unavailable. "
                "Set `NEWSAPI_KEY` environment variable for real headlines."
            )


# ── TAB 3: Evidence ───────────────────────────────────────────────────────────
def render_evidence(risk_summary: dict):
    st.subheader("Exact Filing Evidence for Flagged Risk Categories")
    st.caption(
        "Each passage is drawn directly from the company's 10-K or 10-Q filing. "
        "Sorted by zero-shot classifier confidence."
    )

    flagged = risk_summary["flagged_categories"]

    if not flagged:
        st.success("✅ No risk categories spiked this quarter. No elevated risk evidence to display.")
        return

    st.info(f"**{len(flagged)} flagged:** {', '.join(flagged)}")

    for cat in flagged:
        pct = risk_summary["risk_counts"][cat]["change_pct"]
        curr = risk_summary["risk_counts"][cat]["current_quarter"]
        prev = risk_summary["risk_counts"][cat]["previous_quarter"]

        with st.expander(
            f"⚠️ {cat}  ·  +{pct:.0f}% mentions  ({prev} → {curr} chunks)", expanded=True
        ):
            evidence_list = risk_summary["evidence"].get(cat, [])
            if not evidence_list:
                st.info("No evidence chunks found for this category.")
            else:
                for i, text in enumerate(evidence_list, 1):
                    st.markdown(f"**Evidence {i}**")
                    st.markdown(
                        f"<div style='background:rgba(30,41,59,0.8);border-left:3px solid "
                        f"#3b82f6;padding:12px 16px;border-radius:6px;"
                        f"color:#e2e8f0;line-height:1.6;'>{text[:800]}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")


# ── TAB 4: RAG ───────────────────────────────────────────────────────────────
def render_rag(ticker: str):
    st.subheader("💬 Ask FinRisk")
    st.caption(
        "Questions are answered using only retrieved SEC filing excerpts. "
        "Powered by **Groq Llama-3.3-70b** with cross-encoder reranked hybrid RAG."
    )

    # Suggested questions
    suggestions = [
        f"What are {ticker}'s biggest supply chain risks?",
        f"What cybersecurity threats did {ticker} disclose?",
        f"How does {ticker} describe its liquidity position?",
        f"What regulatory risks is {ticker} facing?",
    ]
    st.markdown("**Suggested questions:**")
    for s in suggestions:
        if st.button(s, key=f"sug_{s[:20]}"):
            st.session_state["rag_query"] = s

    query = st.text_input(
        "Your question",
        value=st.session_state.get("rag_query", ""),
        placeholder=f"What are {ticker}'s biggest supply chain risks?",
        key="rag_input",
    )

    col_btn, col_k = st.columns([2, 1])
    with col_btn:
        ask = st.button("🔍 Ask FinRisk", type="primary", key="ask_btn")
    with col_k:
        top_k = st.selectbox("Chunks to retrieve", [3, 5, 8], index=1, key="rag_k")

    if ask and query.strip():
        with st.spinner("🔍 Retrieving filing excerpts · Generating answer with Groq Llama-3.3-70b…"):
            try:
                from src.rag import FinRiskRAG
                rag = FinRiskRAG()
                result = rag.query(query.strip(), top_k=top_k)
            except Exception as e:
                st.error(f"RAG failed: {e}")
                return

        answer = result.get("answer", {})
        chunks = result.get("retrieved_chunks", [])

        if isinstance(answer, dict):
            summary = answer.get("summary", "No summary available.")
            key_risks = answer.get("key_risks", [])
            confidence = answer.get("confidence", "N/A")
            citations = answer.get("citations", [])
        else:
            summary = str(answer)
            key_risks, confidence, citations = [], "N/A", []

        st.markdown("### Answer")
        st.markdown(
            f"<div style='background:rgba(30,41,59,0.9);border:1px solid #3b82f6;"
            f"border-radius:10px;padding:16px 20px;color:#f1f5f9;line-height:1.7;'>"
            f"{summary}</div>",
            unsafe_allow_html=True,
        )

        if key_risks:
            st.markdown("**Key Risks Identified:**")
            for r in key_risks:
                st.markdown(f"- {r}")

        cols = st.columns(2)
        cols[0].caption(f"Confidence: **{confidence}**")
        if citations:
            cols[1].caption(f"Sources: {', '.join(citations[:3])}")

        with st.expander("📄 Retrieved Source Chunks"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(
                    f"**{i}. {chunk.get('ticker','?')} · "
                    f"{chunk.get('section_type','?')} · "
                    f"reranker score: {chunk.get('reranker_score', 0):.3f}**"
                )
                st.write(chunk.get("text", "")[:400] + "…")
                st.markdown("---")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ticker = render_sidebar()

    st.markdown(
        f"<h1 style='color:#f8fafc;font-weight:700;margin-bottom:0;'>"
        f"{ticker} — Risk Intelligence</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Current quarter vs previous quarter · 10-K + 10-Q · SEC EDGAR")
    st.divider()

    with st.spinner(f"Loading {ticker} data…"):
        risk_summary, sentiment, zscore_history, error = load_company_data(ticker)

    if error:
        st.error(error)
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Risk Changes",
        "💰 Financial & Sentiment",
        "📄 Evidence",
        "💬 Ask FinRisk",
    ])

    with tab1:
        render_risk_changes(risk_summary, ticker)
    with tab2:
        render_financial_sentiment(zscore_history, sentiment, ticker)
    with tab3:
        render_evidence(risk_summary)
    with tab4:
        render_rag(ticker)


if __name__ == "__main__":
    main()
