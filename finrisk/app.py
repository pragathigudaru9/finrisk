"""
FinRisk — Streamlit Dashboard (Stage 9)

5-page research-grade dashboard:
  1. 🏠 Company Overview    — risk gauges, Z-Score, sector view
  2. 🔍 Risk Explorer       — browse & filter retrieved chunks
  3. 📰 Sentiment Trends    — FinBERT 30-day sentiment charts
  4. 🔄 Emerging Risks      — YoY cosine similarity & new risk sentences
  5. 💬 Risk Assistant      — RAG chatbot powered by Groq Llama-3.3-70b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, SENTIMENT_DIR, FINANCIALS_DIR,
    EVAL_DIR, COMPANY_NAMES, SECTOR_MAP, ALL_TICKERS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinRisk — Equity Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a0f1e 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
/* Cards */
.risk-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.risk-high   { border-left: 4px solid #ef4444 !important; }
.risk-elev   { border-left: 4px solid #f59e0b !important; }
.risk-low    { border-left: 4px solid #22c55e !important; }
.chunk-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}
/* Metric cards */
.metric-box {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
h1,h2,h3 { color: #e2e8f0 !important; }
.stMetric label { color: #94a3b8 !important; font-size: 0.8rem; }
.stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
/* Tabs */
[data-baseweb="tab"] { color: #94a3b8; }
[aria-selected="true"] { color: #818cf8 !important; border-bottom-color: #818cf8 !important; }
/* Input */
.stTextInput input, .stSelectbox select, .stMultiSelect [data-baseweb="select"] {
    background: rgba(30,41,59,0.8) !important;
    color: #e2e8f0 !important;
    border-color: rgba(99,102,241,0.3) !important;
}
/* Button */
.stButton > button { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white;
    border: none; border-radius: 8px; font-weight: 600; }
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

RISK_EMOJI = {"High": "🔴", "Elevated": "🟡", "Low": "🟢"}
RISK_COLOR = {"High": "#ef4444", "Elevated": "#f59e0b", "Low": "#22c55e"}
INDIGO = "#818cf8"
PLOTLY_TEMPLATE = "plotly_dark"

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_risk():
    p = PROCESSED_DIR / "risk_scores.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_sentiment():
    p = SENTIMENT_DIR / "sentiment_scores.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_zscore():
    p = FINANCIALS_DIR / "zscore.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_trends():
    p = PROCESSED_DIR / "yoy_trends.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    for col in ["new_risks", "removed_risks", "changed_risks"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

@st.cache_data
def load_eval():
    ab_p, rg_p = EVAL_DIR / "ablation_results.json", EVAL_DIR / "ragas_results.json"
    ab = json.loads(ab_p.read_text()) if ab_p.exists() else {}
    rg = json.loads(rg_p.read_text()) if rg_p.exists() else {}
    return ab, rg

@st.cache_resource
def load_rag():
    try:
        from src.rag import FinRiskRAG
        return FinRiskRAG()
    except Exception as e:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 FinRisk")
        st.markdown("<small style='color:#94a3b8'>Equity Risk Intelligence System</small>", unsafe_allow_html=True)
        st.divider()

        risk_df = load_risk()
        if not risk_df.empty:
            total = len(risk_df)
            high = len(risk_df[risk_df["risk_label"] == "High"])
            elev = len(risk_df[risk_df["risk_label"] == "Elevated"])
            low = len(risk_df[risk_df["risk_label"] == "Low"])
            st.markdown("**Portfolio Summary**")
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 High", high)
            c2.metric("🟡 Elev.", elev)
            c3.metric("🟢 Low", low)
            st.markdown(f"<small style='color:#64748b'>{total} companies tracked</small>", unsafe_allow_html=True)

        st.divider()
        ab, rg = load_eval()
        if ab:
            st.markdown("**Retrieval Ablation**")
            rows = []
            for method, m in ab.items():
                rows.append({"Method": method, "Hit@5": m.get("Hit@5","?"), "MRR": m.get("MRR","?")})
            st.dataframe(pd.DataFrame(rows).set_index("Method"), use_container_width=True)
        if rg and "summary" in rg:
            st.markdown("**RAG Quality (RAGAS)**")
            s = rg["summary"]
            c1, c2 = st.columns(2)
            c1.metric("Faithfulness", f"{s.get('faithfulness',0):.2f}")
            c2.metric("Relevance", f"{s.get('answer_relevance',0):.2f}")

        st.divider()
        st.markdown("<small style='color:#475569'>Built with Hybrid RAG · BGE · FinBERT · Groq Llama-3.3-70b</small>", unsafe_allow_html=True)

# ── PAGE 1: Company Overview ───────────────────────────────────────────────────
def page_overview():
    st.markdown("## 🏠 Company Overview")
    st.markdown("Select a company to view its complete risk profile.")

    risk_df = load_risk()
    if risk_df.empty:
        st.warning("⚠️ Risk scores not available. Please run the pipeline first.")
        return

    zscore_df = load_zscore()
    sentiment_df = load_sentiment()

    available = sorted(risk_df["ticker"].unique())
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Select Company", available,
                              format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t, t)}",
                              key="overview_ticker")
    with col2:
        sector = SECTOR_MAP.get(ticker, "Unknown")
        st.markdown(f"<div class='metric-box'><small style='color:#94a3b8'>Sector</small><br><b>{sector}</b></div>",
                    unsafe_allow_html=True)

    row = risk_df[risk_df["ticker"] == ticker].iloc[0]
    label = row["risk_label"]
    score = row["risk_score"]
    color = RISK_COLOR[label]
    emoji = RISK_EMOJI[label]

    st.markdown("---")

    # Top metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Composite Risk Score", f"{score:.1f}/100")
    mc2.metric("Risk Label", f"{emoji} {label}")
    mc3.metric("Filing Risk", f"{row['filing_component']:.0f}/100")

    z_row = zscore_df[zscore_df["ticker"] == ticker] if not zscore_df.empty else pd.DataFrame()
    if not z_row.empty:
        z = z_row.iloc[0]
        mc4.metric("Altman Z-Score", f"{z['z_score']:.2f} ({z['zone']})")
    else:
        mc4.metric("Altman Z-Score", "N/A")

    # Gauge + Breakdown
    col_g, col_b = st.columns(2)

    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Risk Score — {ticker}", "font": {"color": "#e2e8f0", "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 33], "color": "rgba(34,197,94,0.15)"},
                    {"range": [33, 66], "color": "rgba(245,158,11,0.15)"},
                    {"range": [66, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "value": score},
                "bgcolor": "rgba(0,0,0,0)",
            },
            number={"suffix": "", "font": {"color": "#e2e8f0", "size": 28}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"}, height=280, margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_b:
        components = {
            "Filing Risk\n(40%)": row["filing_component"],
            "Sentiment Risk\n(35%)": row["sentiment_component"],
            "Z-Score Risk\n(25%)": row["zscore_component"],
        }
        fig_bar = go.Figure(go.Bar(
            x=list(components.keys()),
            y=list(components.values()),
            marker_color=[INDIGO, "#06b6d4", "#22c55e"],
            text=[f"{v:.0f}" for v in components.values()],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title={"text": "Risk Component Breakdown", "font": {"color": "#e2e8f0"}},
            yaxis=dict(range=[0, 105], tickcolor="#64748b", gridcolor="rgba(100,116,139,0.2)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"}, height=280, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Sector leaderboard
    st.markdown("### 📊 Sector Risk Leaderboard")
    sector_data = risk_df.copy()
    sector_data["Emoji"] = sector_data["risk_label"].map(RISK_EMOJI)
    sector_data["Display"] = sector_data["Emoji"] + " " + sector_data["risk_label"]
    col_config = {
        "ticker": st.column_config.TextColumn("Ticker", width=80),
        "company_name": st.column_config.TextColumn("Company", width=200),
        "sector": st.column_config.TextColumn("Sector", width=120),
        "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
        "Display": st.column_config.TextColumn("Label", width=110),
    }
    highlight = risk_df[risk_df["ticker"] == ticker].index[0] if ticker in risk_df["ticker"].values else None
    st.dataframe(
        sector_data[["ticker", "company_name", "sector", "risk_score", "Display"]].rename(columns={"Display": "Risk Label"}),
        column_config=col_config,
        use_container_width=True, hide_index=True,
    )

    # Sector heatmap
    st.markdown("### 🗺️ Sector Risk Heatmap")
    sector_avg = risk_df.groupby("sector")["risk_score"].mean().reset_index()
    sector_count = risk_df.groupby("sector")["ticker"].count().reset_index().rename(columns={"ticker": "count"})
    sector_merged = sector_avg.merge(sector_count, on="sector")
    fig_sec = px.bar(
        sector_merged.sort_values("risk_score", ascending=True),
        x="risk_score", y="sector", orientation="h",
        color="risk_score", color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        text="count", title="Average Risk Score by Sector",
        template=PLOTLY_TEMPLATE,
    )
    fig_sec.update_traces(texttemplate="%{text} co.", textposition="outside")
    fig_sec.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}, height=320, margin=dict(l=20, r=40, t=50, b=20),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_sec, use_container_width=True)

# ── PAGE 2: Risk Explorer ─────────────────────────────────────────────────────
def page_risk_explorer():
    st.markdown("## 🔍 Risk Explorer")
    st.markdown("Browse, filter, and explore SEC filing excerpts from the vector index.")

    rag = load_rag()
    if rag is None:
        st.error("RAG pipeline unavailable. Run the full pipeline first.")
        return

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Search query", placeholder="e.g. supply chain disruption risks",
                               key="explorer_query")
    with col2:
        top_k = st.selectbox("Top K", [5, 10, 15], index=0, key="explorer_k")
    with col3:
        section_filter = st.selectbox("Section", ["All", "Risk Factors", "MD&A", "Financial Statements"],
                                       key="explorer_section")

    if query:
        with st.spinner("Retrieving chunks..."):
            result = rag.retriever.retrieve(query, top_k=top_k)

        if section_filter != "All":
            result = [r for r in result if r["section_type"] == section_filter]

        st.markdown(f"**{len(result)} excerpts retrieved** · Strategy: `section_aware`")

        for i, chunk in enumerate(result):
            label_color = {"Risk Factors": "#ef4444", "MD&A": "#06b6d4",
                           "Financial Statements": "#22c55e"}.get(chunk["section_type"], INDIGO)
            score_bar = min(chunk.get("reranker_score", 0) * 10 + 50, 100)

            with st.expander(
                f"[{i+1}] **{chunk['ticker']}** · {chunk['year']} · {chunk['section_type']} "
                f"· Reranker: `{chunk.get('reranker_score', 0):.4f}`"
            ):
                cols = st.columns([2, 1])
                with cols[0]:
                    st.markdown(f"<span style='background:rgba({','.join(str(int(label_color.lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.2);padding:2px 8px;border-radius:4px;font-size:0.8rem'>{chunk['section_type']}</span>", unsafe_allow_html=True)
                    st.markdown(chunk["text"][:800])
                with cols[1]:
                    st.metric("FAISS Score", f"{chunk.get('faiss_score', 0):.4f}")
                    st.metric("BM25 Score", f"{chunk.get('bm25_score', 0):.4f}")
                    st.metric("Source", chunk.get("source", "hybrid"))
                    st.caption(f"Chunk ID: `{chunk['chunk_id'][:40]}`")

# ── PAGE 3: Sentiment Trends ──────────────────────────────────────────────────
def page_sentiment():
    st.markdown("## 📰 Sentiment Trends")
    st.markdown("FinBERT-scored financial news sentiment (30-day rolling averages).")

    sentiment_df = load_sentiment()
    if sentiment_df.empty:
        st.warning("No sentiment data. Run `python -m src.sentiment` first.")
        return

    available = sorted(sentiment_df["ticker"].unique())
    selected = st.multiselect(
        "Select companies to compare",
        available,
        default=["AAPL", "TSLA", "NVDA", "JPM", "BA"][:min(5, len(available))],
        format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t, t)}",
        key="sentiment_tickers",
    )

    if not selected:
        st.info("Select at least one company.")
        return

    # 30-day rolling comparison chart
    filtered = sentiment_df[sentiment_df["ticker"].isin(selected)].copy()
    filtered["date"] = pd.to_datetime(filtered["date"])

    fig = go.Figure()
    palette = [INDIGO, "#06b6d4", "#22c55e", "#f59e0b", "#ef4444",
               "#a855f7", "#ec4899", "#14b8a6"]
    for i, t in enumerate(selected):
        td = filtered[filtered["ticker"] == t].sort_values("date")
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=td["date"], y=td["rolling_30d"],
            name=f"{t} (30d)", mode="lines", line=dict(color=color, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=td["date"], y=td["daily_sentiment"],
            name=f"{t} (daily)", mode="lines",
            line=dict(color=color, width=1, dash="dot"), opacity=0.35,
        ))

    fig.add_hline(y=0, line=dict(color="#64748b", dash="dash", width=1))
    fig.update_layout(
        title="30-Day Rolling Sentiment (FinBERT)",
        xaxis_title="Date", yaxis_title="Sentiment Score [-1, +1]",
        yaxis=dict(range=[-1, 1], gridcolor="rgba(100,116,139,0.2)"),
        xaxis=dict(gridcolor="rgba(100,116,139,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}, legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=420, template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Latest sentiment leaderboard
    st.markdown("### Latest Sentiment Snapshot")
    latest = sentiment_df.groupby("ticker").last().reset_index()
    latest["company"] = latest["ticker"].map(COMPANY_NAMES)
    latest["Mood"] = latest["rolling_30d"].apply(
        lambda x: "🟢 Positive" if x > 0.05 else ("🔴 Negative" if x < -0.05 else "🟡 Neutral")
    )
    latest_sorted = latest.sort_values("rolling_30d", ascending=False)
    col_cfg = {
        "ticker": st.column_config.TextColumn("Ticker", width=80),
        "company": st.column_config.TextColumn("Company", width=200),
        "rolling_30d": st.column_config.NumberColumn("30d Avg", format="%.4f"),
        "rolling_7d": st.column_config.NumberColumn("7d Avg", format="%.4f"),
        "Mood": st.column_config.TextColumn("Mood"),
    }
    st.dataframe(
        latest_sorted[["ticker", "company", "rolling_30d", "rolling_7d", "Mood"]],
        column_config=col_cfg, use_container_width=True, hide_index=True,
    )

    # Bearish flags
    n_flags = sentiment_df["sentiment_flag"].sum()
    if n_flags > 0:
        st.markdown(f"⚠️ **{int(n_flags)} bearish flags** detected (7d avg > 0.15 below 30d avg)")

# ── PAGE 4: Emerging Risks ────────────────────────────────────────────────────
def page_emerging():
    st.markdown("## 🔄 Emerging Risk Detection")
    st.markdown("Cosine similarity between consecutive years' Risk Factor sections. "
                "Low similarity → significant risk language change.")

    trends_df = load_trends()
    if trends_df.empty:
        st.warning("No YoY trend data. Run `python -m src.risk_score` first.")
        return

    # Similarity heatmap
    st.markdown("### Cosine Similarity Heatmap")
    pivot_data = []
    for _, row in trends_df.iterrows():
        pivot_data.append({
            "ticker": row["ticker"],
            "year_pair": row["year_pair"],
            "cosine_similarity": row["cosine_similarity"],
        })
    pivot_df = pd.DataFrame(pivot_data)

    if not pivot_df.empty and len(pivot_df["year_pair"].unique()) > 0:
        try:
            pivot_table = pivot_df.pivot(index="ticker", columns="year_pair", values="cosine_similarity")
            fig_heat = px.imshow(
                pivot_table,
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                zmin=0.5, zmax=1.0,
                title="Risk Language Similarity (1.0 = identical, <0.85 = significant change)",
                template=PLOTLY_TEMPLATE,
                text_auto=".3f",
            )
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"}, height=max(400, len(pivot_table) * 22 + 100),
                margin=dict(l=60, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            st.dataframe(pivot_df, use_container_width=True)

    # Flagged companies
    changed = trends_df[trends_df["change_type"] == "Risk Language Changed"].copy()
    st.markdown(f"### ⚠️ {len(changed)} Year-Pairs Flagged for Significant Risk Changes")

    if changed.empty:
        st.success("No significant risk language changes detected.")
        return

    changed_sorted = changed.sort_values("trend_score", ascending=False).reset_index(drop=True)

    for _, row in changed_sorted.iterrows():
        new = row.get("new_risks", []) or []
        removed = row.get("removed_risks", []) or []
        if isinstance(new, str):
            new = json.loads(new)
        if isinstance(removed, str):
            removed = json.loads(removed)

        with st.expander(
            f"⚠️ **{row['ticker']}** · {row['year_pair']} · "
            f"Similarity: {row['cosine_similarity']:.4f} · "
            f"Trend Score: {row['trend_score']:.1f}"
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🆕 Newly Emerging Risks:**")
                if new:
                    for s in new[:3]:
                        st.markdown(f"<div class='chunk-card'><span style='color:#ef4444'>▶ NEW</span><br>{s[:250]}</div>",
                                    unsafe_allow_html=True)
                else:
                    st.caption("None detected")
            with c2:
                st.markdown("**🗑️ Reduced/Removed Risks:**")
                if removed:
                    for s in removed[:3]:
                        st.markdown(f"<div class='chunk-card'><span style='color:#94a3b8'>◀ OLD</span><br>{s[:250]}</div>",
                                    unsafe_allow_html=True)
                else:
                    st.caption("None detected")

    # Bar chart: trend score
    fig_trend = px.bar(
        changed_sorted,
        x="ticker", y="trend_score", color="trend_score",
        color_continuous_scale=["#f59e0b", "#ef4444"],
        title="Trend Score by Company (higher = more risk language change)",
        template=PLOTLY_TEMPLATE, text="year_pair",
    )
    fig_trend.update_traces(textposition="outside")
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}, height=350,
        coloraxis_showscale=False, margin=dict(l=20, r=20, t=50, b=60),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ── PAGE 5: Risk Assistant ──────────────────────────────────────────────────────
def page_assistant():
    st.markdown("## 💬 Risk Assistant")
    st.markdown("Ask any question about company risks, SEC filings, or financial health. "
                "Powered by citation-backed hybrid RAG.")

    rag = load_rag()
    if rag is None:
        st.error("RAG pipeline not available. Run the full pipeline first.")
        return

    # Example questions
    examples = [
        "What are Apple's biggest supply chain risks?",
        "How does Tesla describe competition in the EV market?",
        "What credit risks did JPMorgan disclose?",
        "What new risks did NVIDIA face in 2024?",
        "How do energy companies discuss climate change risk?",
        "What cybersecurity threats does Microsoft mention?",
    ]

    with st.expander("💡 Example Questions"):
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=False):
                st.session_state["rag_q"] = ex

    question = st.text_area(
        "Your question",
        value=st.session_state.get("rag_q", ""),
        placeholder="e.g. What are Tesla's biggest risks in 2024?",
        height=80, key="rag_question",
    )

    col_btn, col_k = st.columns([2, 1])
    with col_btn:
        ask = st.button("🔍 Ask FinRisk", type="primary", key="ask_btn")
    with col_k:
        top_k = st.selectbox("Chunks", [3, 5, 8], index=1, key="rag_k")

    if ask and question.strip():
        with st.spinner("🔍 Retrieving from 15,000+ chunks · Generating answer with Groq Llama-3.3-70b..."):
            result = rag.query(question.strip(), top_k=top_k)

        answer = result["answer"]
        chunks = result["retrieved_chunks"]

        # Answer card
        confidence = answer.get("confidence", "medium")
        conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(confidence, "#94a3b8")
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")

        st.markdown("---")
        st.markdown("### 📝 Answer")

        st.markdown(
            f"<div class='risk-card'>"
            f"<p style='font-size:1.05rem;line-height:1.6'>{answer.get('summary', 'No summary.')}</p>"
            f"<small style='color:{conf_color}'>{conf_emoji} Confidence: {confidence}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_r, col_c = st.columns(2)
        with col_r:
            key_risks = answer.get("key_risks", [])
            if key_risks:
                st.markdown("**⚡ Key Risks Identified:**")
                for risk in key_risks:
                    st.markdown(f"• {risk[:200]}")

        with col_c:
            citations = answer.get("citations", [])
            if citations:
                st.markdown("**📌 Citations:**")
                for cit in citations:
                    st.markdown(
                        f"<div style='background:rgba(99,102,241,0.1);border-left:3px solid {INDIGO};"
                        f"padding:4px 10px;border-radius:4px;margin:4px 0;font-size:0.85rem'>"
                        f"<b>[{cit.get('source', '?')}]</b> {cit.get('claim', '')[:150]}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Retrieved chunks
        with st.expander(f"📚 View {len(chunks)} Retrieved Excerpts"):
            for i, chunk in enumerate(chunks):
                score_color = "#22c55e" if chunk.get("reranker_score", 0) > 0.5 else "#f59e0b"
                st.markdown(
                    f"<div class='chunk-card'>"
                    f"<b style='color:{INDIGO}'>[{i+1}] {chunk.get('ticker','?')} · "
                    f"{chunk.get('year','?')} · {chunk.get('section_type','?')}</b> "
                    f"<span style='color:{score_color};float:right'>Reranker: {chunk.get('reranker_score',0):.4f}</span><br>"
                    f"<small style='color:#94a3b8'>{chunk.get('text','')[:400]}...</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Company Overview",
        "🔍 Risk Explorer",
        "📰 Sentiment Trends",
        "🔄 Emerging Risks",
        "💬 Risk Assistant",
    ])

    with tab1:
        page_overview()
    with tab2:
        page_risk_explorer()
    with tab3:
        page_sentiment()
    with tab4:
        page_emerging()
    with tab5:
        page_assistant()


if __name__ == "__main__":
    main()
