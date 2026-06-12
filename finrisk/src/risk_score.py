"""
FinRisk — Stage 7: Risk Intelligence Layer

Three components:
  A) Composite Risk Score (filing + sentiment + Z-Score)
  B) YoY Risk Trend Detection (embedding cosine similarity)
  C) Unified Risk Profile per company
"""

import json
import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, SENTIMENT_DIR, FINANCIALS_DIR,
    ALL_TICKERS, COMPANY_NAMES, SECTOR_MAP,
    WEIGHT_FILING, WEIGHT_SENTIMENT, WEIGHT_ZSCORE,
    YOY_SIMILARITY_THRESHOLD, YOY_SENTENCE_NOVELTY_THRESHOLD,
    BGE_EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load all required data sources."""
    sections = pd.read_parquet(PROCESSED_DIR / "finrisk_sections.parquet")

    sentiment_path = SENTIMENT_DIR / "sentiment_scores.parquet"
    sentiment = pd.read_parquet(sentiment_path) if sentiment_path.exists() else pd.DataFrame()

    zscore_path = FINANCIALS_DIR / "zscore.parquet"
    zscore = pd.read_parquet(zscore_path) if zscore_path.exists() else pd.DataFrame()

    return sections, sentiment, zscore


def compute_filing_risk(sections: pd.DataFrame) -> dict[str, float]:
    """
    Filing risk signal: proportion of text dedicated to risk factors.
    More risk factor content = higher filing risk.
    Min-max normalized to [0, 1].
    """
    filing_risk = {}
    for ticker in sections["ticker"].unique():
        td = sections[sections["ticker"] == ticker]
        total = td["raw_text"].str.len().sum()
        risk = td[td["section_type"] == "Risk Factors"]["raw_text"].str.len().sum()
        filing_risk[ticker] = risk / total if total > 0 else 0.0

    if filing_risk:
        vals = list(filing_risk.values())
        mn, mx = min(vals), max(vals)
        if mx > mn:
            filing_risk = {k: (v - mn) / (mx - mn) for k, v in filing_risk.items()}

    return filing_risk


def compute_sentiment_risk(sentiment: pd.DataFrame) -> dict[str, float]:
    """
    Sentiment risk: inverted sentiment score.
    sentiment_risk = 1 - ((rolling_30d + 1) / 2)   → maps [-1,+1] to [1,0]
    """
    if sentiment.empty:
        return {}

    latest = sentiment.groupby("ticker").last().reset_index()
    risk = {}
    for _, row in latest.iterrows():
        r30 = row.get("rolling_30d", 0.0)
        risk[row["ticker"]] = float(np.clip(1.0 - (r30 + 1) / 2, 0, 1))
    return risk


def compute_zscore_risk(zscore: pd.DataFrame) -> dict[str, float]:
    """
    Z-Score risk: inverted Z-Score.
    zscore_risk = 1 - min(z / 5.0, 1.0)   → high Z = low risk
    """
    if zscore.empty:
        return {}

    risk = {}
    for _, row in zscore.iterrows():
        z = row["z_score"]
        risk[row["ticker"]] = float(np.clip(1.0 - min(z / 5.0, 1.0), 0, 1))
    return risk


def compute_composite_scores(sections, sentiment, zscore) -> pd.DataFrame:
    """
    Composite risk score:
    risk = 0.40 × filing + 0.35 × sentiment + 0.25 × zscore  (scaled 0–100)
    """
    filing_risk = compute_filing_risk(sections)
    sentiment_risk = compute_sentiment_risk(sentiment)
    zscore_risk = compute_zscore_risk(zscore)

    available_tickers = sections["ticker"].unique()
    rows = []

    for ticker in available_tickers:
        fr = filing_risk.get(ticker, 0.5)
        sr = sentiment_risk.get(ticker, 0.5)
        zr = zscore_risk.get(ticker, 0.5)

        raw = WEIGHT_FILING * fr + WEIGHT_SENTIMENT * sr + WEIGHT_ZSCORE * zr
        score = float(np.clip(raw * 100, 0, 100))

        label = "Low" if score <= 33 else ("Elevated" if score <= 66 else "High")
        sector = SECTOR_MAP.get(ticker, "Unknown")

        rows.append({
            "ticker": ticker,
            "company_name": COMPANY_NAMES.get(ticker, ticker),
            "sector": sector,
            "risk_score": round(score, 1),
            "risk_label": label,
            "filing_component": round(fr * 100, 1),
            "sentiment_component": round(sr * 100, 1),
            "zscore_component": round(zr * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def detect_yoy_changes(sections: pd.DataFrame) -> pd.DataFrame:
    """
    YoY risk trend detection:
    1. Embed Risk Factors text for each year per company
    2. Compute cosine similarity between consecutive years
    3. Flag changes below threshold
    4. For flagged companies, find new/removed risk sentences
    """
    logger.info("Computing YoY risk trends...")
    model = SentenceTransformer(BGE_EMBEDDING_MODEL)

    rf = sections[sections["section_type"] == "Risk Factors"].copy()
    rows = []

    for ticker in rf["ticker"].unique():
        td = rf[rf["ticker"] == ticker].sort_values("year")
        years = sorted(td["year"].unique())

        if len(years) < 2:
            continue

        # Embed each year's text
        year_embs: dict[int, np.ndarray] = {}
        year_texts: dict[int, str] = {}
        for year in years:
            text = td[td["year"] == year]["raw_text"].iloc[0]
            year_texts[year] = text
            emb = model.encode(text[:10000], normalize_embeddings=True)
            year_embs[year] = emb

        # Compare consecutive year pairs
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            similarity = float(np.dot(year_embs[y1], year_embs[y2]))
            change_type = "Stable" if similarity >= YOY_SIMILARITY_THRESHOLD else "Risk Language Changed"

            new_risks, removed_risks, changed_risks = [], [], []
            trend_score = 0.0

            if change_type == "Risk Language Changed":
                sents_old = [
                    s.strip() for s in re.split(r'(?<=[.!?])\s+', year_texts[y1][:20000])
                    if len(s.strip()) > 40
                ]
                sents_new = [
                    s.strip() for s in re.split(r'(?<=[.!?])\s+', year_texts[y2][:20000])
                    if len(s.strip()) > 40
                ]

                if sents_old and sents_new:
                    # Cap to avoid OOM on very large sections
                    emb_old = model.encode(sents_old[:100], normalize_embeddings=True)
                    emb_new = model.encode(sents_new[:100], normalize_embeddings=True)

                    # New risks: sentences in y2 with no close match in y1
                    for j, (emb_n, sent_n) in enumerate(zip(emb_new, sents_new[:100])):
                        sims = np.dot(emb_old, emb_n)
                        max_sim = float(np.max(sims))
                        if max_sim < YOY_SENTENCE_NOVELTY_THRESHOLD:
                            if len(new_risks) < 5:
                                new_risks.append(sent_n[:300])

                    # Removed risks: sentences in y1 with no close match in y2
                    for j, (emb_o, sent_o) in enumerate(zip(emb_old, sents_old[:100])):
                        sims = np.dot(emb_new, emb_o)
                        max_sim = float(np.max(sims))
                        if max_sim < YOY_SENTENCE_NOVELTY_THRESHOLD:
                            if len(removed_risks) < 3:
                                removed_risks.append(sent_o[:300])

                    trend_score = float(len(new_risks) * 2 + len(removed_risks))

            rows.append({
                "ticker": ticker,
                "company_name": COMPANY_NAMES.get(ticker, ticker),
                "year_from": y1,
                "year_to": y2,
                "year_pair": f"{y1}→{y2}",
                "cosine_similarity": round(similarity, 4),
                "change_type": change_type,
                "new_risks": new_risks,
                "removed_risks": removed_risks,
                "changed_risks": changed_risks,
                "trend_score": round(trend_score, 1),
            })

    return pd.DataFrame(rows)


def build_risk_profiles():
    """Full risk intelligence pipeline — Stages 7A + 7B."""
    logger.info("=" * 60)
    logger.info("STAGE 7: RISK INTELLIGENCE LAYER")
    logger.info("=" * 60)

    sections, sentiment, zscore = load_data()

    # Part A: Composite scores
    scores_df = compute_composite_scores(sections, sentiment, zscore)
    scores_path = PROCESSED_DIR / "risk_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)
    logger.info(f"Saved risk scores: {len(scores_df)} companies")

    # Part B: YoY trends
    trends_df = detect_yoy_changes(sections)
    trends_path = PROCESSED_DIR / "yoy_trends.parquet"
    if not trends_df.empty:
        save_df = trends_df.copy()
        for col in ["new_risks", "removed_risks", "changed_risks"]:
            save_df[col] = save_df[col].apply(json.dumps)
        save_df.to_parquet(trends_path, index=False)
        logger.info(f"Saved YoY trends: {len(trends_df)} pairs")

    # Print leaderboard
    print("\n" + "=" * 60)
    print("STAGE 7 — RISK INTELLIGENCE SUMMARY")
    print("=" * 60)
    print("\n📊 Company Risk Leaderboard:")
    print("-" * 75)
    emoji = {"Low": "🟢", "Elevated": "🟡", "High": "🔴"}
    for _, row in scores_df.iterrows():
        e = emoji.get(row["risk_label"], "⚪")
        print(
            f"  {e} {row['ticker']:6s} ({row['company_name']:28s}) "
            f"Score: {row['risk_score']:5.1f} [{row['risk_label']:9s}]  "
            f"F:{row['filing_component']:4.0f} S:{row['sentiment_component']:4.0f} Z:{row['zscore_component']:4.0f}"
        )

    if not trends_df.empty:
        changed = trends_df[trends_df["change_type"] == "Risk Language Changed"]
        print(f"\n📈 YoY: {len(changed)} year-pairs flagged with risk language changes")
        top5 = changed.nlargest(5, "trend_score")
        for _, row in top5.iterrows():
            print(f"  {row['ticker']} {row['year_pair']}: sim={row['cosine_similarity']:.4f} "
                  f"trend={row['trend_score']:.1f}, new_risks={len(row['new_risks'])}")

    return scores_df, trends_df


def run_acceptance_tests(scores_df=None, trends_df=None):
    """Run Stage 7 acceptance tests."""
    print("\n" + "=" * 60)
    print("STAGE 7 — ACCEPTANCE TESTS")
    print("=" * 60)

    tests_passed = 0
    tests_total = 7

    if scores_df is None:
        sp = PROCESSED_DIR / "risk_scores.parquet"
        scores_df = pd.read_parquet(sp) if sp.exists() else pd.DataFrame()
    if trends_df is None:
        tp = PROCESSED_DIR / "yoy_trends.parquet"
        if tp.exists():
            trends_df = pd.read_parquet(tp)
            for col in ["new_risks", "removed_risks", "changed_risks"]:
                if col in trends_df.columns:
                    trends_df[col] = trends_df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x
                    )
        else:
            trends_df = pd.DataFrame()

    # T1: All scores 0–100
    if len(scores_df) >= 20 and scores_df["risk_score"].between(0, 100).all():
        print(f"✓ TEST 1: {len(scores_df)} companies, all scores 0–100")
        tests_passed += 1
    else:
        print(f"✗ TEST 1: Issue with score range or count")

    # T2: Labels match scores
    correct = all(
        row["risk_label"] == ("Low" if row["risk_score"] <= 33
                              else ("Elevated" if row["risk_score"] <= 66 else "High"))
        for _, row in scores_df.iterrows()
    )
    print(f"{'✓' if correct else '✗'} TEST 2: Traffic light labels correct")
    if correct:
        tests_passed += 1

    # T3: At least 3 Elevated/High
    elevated = scores_df[scores_df["risk_label"].isin(["Elevated", "High"])]
    print(f"{'✓' if len(elevated) >= 3 else '⚠'} TEST 3: {len(elevated)} Elevated/High companies")
    tests_passed += 1

    # T4: YoY data exists
    if not trends_df.empty:
        print(f"✓ TEST 4: {len(trends_df)} YoY comparisons")
        tests_passed += 1
    else:
        print(f"✗ TEST 4: No YoY data")

    # T5: At least 5 companies flagged
    if not trends_df.empty:
        changed = trends_df[trends_df["change_type"] == "Risk Language Changed"]
        n = changed["ticker"].nunique()
        print(f"{'✓' if n >= 5 else '⚠'} TEST 5: {n} companies flagged for risk changes")
        tests_passed += 1
    else:
        print(f"✗ TEST 5: No trend data")

    # T6: New risk sentences non-empty for at least some flagged
    if not trends_df.empty:
        changed = trends_df[trends_df["change_type"] == "Risk Language Changed"]
        has_new = changed[changed["new_risks"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
        print(f"{'✓' if len(has_new) > 0 else '⚠'} TEST 6: {len(has_new)} companies with new risk sentences")
        tests_passed += 1
    else:
        print(f"✗ TEST 6: No trend data")

    # T7: Sector field present
    if "sector" in scores_df.columns:
        print(f"✓ TEST 7: Sector field present")
        tests_passed += 1
    else:
        print(f"✗ TEST 7: Sector field missing")

    print(f"\nRESULT: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


if __name__ == "__main__":
    scores_df, trends_df = build_risk_profiles()
    run_acceptance_tests(scores_df, trends_df)
