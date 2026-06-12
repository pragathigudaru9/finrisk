"""
FinRisk — Stage 6A: FinBERT Sentiment Pipeline

Scores news headlines for each company using FinBERT.
Computes daily, 7-day, and 30-day rolling sentiment averages.
Falls back to realistic synthetic data if NEWSAPI_KEY is not set.
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALL_TICKERS, COMPANY_NAMES, SENTIMENT_DIR,
    NEWSAPI_KEY, FINBERT_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logger.info(f"Using device: {DEVICE}")


def fetch_news_real(ticker: str, company_name: str) -> list[dict]:
    """Fetch headlines from NewsAPI."""
    from newsapi import NewsApiClient

    api = NewsApiClient(api_key=NEWSAPI_KEY)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    try:
        response = api.get_everything(
            q=company_name,
            language="en",
            sort_by="publishedAt",
            from_param=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            page_size=50,
        )
        articles = response.get("articles", [])
        headlines = []
        for art in articles:
            title = art.get("title", "")
            if title and title != "[Removed]":
                headlines.append({
                    "ticker": ticker,
                    "date": art.get("publishedAt", "")[:10],
                    "headline": title,
                })
        logger.info(f"  {ticker}: fetched {len(headlines)} headlines from NewsAPI")
        return headlines
    except Exception as e:
        logger.warning(f"NewsAPI error for {ticker}: {e}")
        return []


# Realistic sentiment seeds per sector
SECTOR_SEEDS = {
    "Technology": 0.15,
    "Financials": 0.05,
    "Healthcare": 0.10,
    "Energy": -0.05,
    "Consumer Staples": 0.08,
    "Consumer Discretionary": 0.12,
    "Industrials": 0.02,
}

TICKER_BASE_SENTIMENT = {
    "AAPL": 0.18, "MSFT": 0.20, "GOOGL": 0.12, "NVDA": 0.25, "TSLA": 0.05,
    "META": 0.10, "AMD": 0.15, "INTC": -0.05,
    "JPM": 0.10, "BAC": 0.06, "GS": 0.08, "MS": 0.07, "C": 0.02,
    "JNJ": 0.08, "PFE": -0.02, "MRK": 0.12, "ABBV": 0.15, "LLY": 0.22,
    "XOM": 0.05, "CVX": 0.04, "COP": 0.03,
    "WMT": 0.12, "PG": 0.09, "KO": 0.10,
    "BA": -0.08, "CAT": 0.06, "GE": 0.08,
    "DAL": 0.04, "AAL": -0.05,
}


def generate_synthetic_headlines(ticker: str) -> list[dict]:
    """Generate realistic synthetic sentiment data for a ticker."""
    base = TICKER_BASE_SENTIMENT.get(ticker, 0.0)
    rng = np.random.default_rng(seed=abs(hash(ticker)) % (2**31))

    end_date = datetime.now()
    headlines = []

    for day_offset in range(30):
        date = end_date - timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        n_headlines = rng.integers(2, 6)

        for _ in range(n_headlines):
            # Noise around base sentiment with occasional shock events
            shock = rng.choice([0.0, 0.3, -0.3], p=[0.85, 0.075, 0.075])
            score = float(np.clip(rng.normal(base + shock, 0.18), -1.0, 1.0))
            headlines.append({
                "ticker": ticker,
                "date": date_str,
                "headline": f"[Synthetic] {COMPANY_NAMES.get(ticker, ticker)} news",
                "synthetic_score": score,
            })

    return headlines


def score_with_finbert(headlines: list[str], tokenizer, model) -> list[float]:
    """Score headlines using FinBERT. Returns sentiment scores [-1, 1]."""
    if not headlines:
        return []

    scores = []
    batch_size = 16

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        inputs = tokenizer(
            batch,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # FinBERT labels: [positive, negative, neutral]
        for prob in probs:
            p_pos = prob[0].item()
            p_neg = prob[1].item()
            scores.append(p_pos - p_neg)

    return scores


def build_sentiment_parquet():
    """
    Full sentiment pipeline:
    1. Fetch headlines (real or synthetic)
    2. Score with FinBERT (if real)
    3. Compute rolling averages (7d, 30d)
    4. Flag sentiment drops
    5. Save parquet
    """
    logger.info("=" * 60)
    logger.info("STAGE 6A: FINBERT SENTIMENT PIPELINE")
    logger.info("=" * 60)

    use_real = bool(NEWSAPI_KEY)
    logger.info(f"Mode: {'NewsAPI (real)' if use_real else 'Synthetic (no NEWSAPI_KEY)'}")

    tokenizer = None
    model = None
    if use_real:
        logger.info(f"Loading FinBERT: {FINBERT_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        model.to(DEVICE)
        model.eval()

    all_rows = []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        company_name = COMPANY_NAMES.get(ticker, ticker)
        logger.info(f"[{i}/{len(ALL_TICKERS)}] {ticker} ({company_name})")

        if use_real:
            headlines_data = fetch_news_real(ticker, company_name)
            if not headlines_data:
                logger.warning(f"  No headlines — using synthetic")
                headlines_data = generate_synthetic_headlines(ticker)
        else:
            headlines_data = generate_synthetic_headlines(ticker)

        # Group by date
        date_groups: dict[str, list] = {}
        for h in headlines_data:
            date_groups.setdefault(h["date"], []).append(h)

        for date, items in sorted(date_groups.items()):
            if use_real and "synthetic_score" not in items[0]:
                texts = [h["headline"] for h in items]
                scores = score_with_finbert(texts, tokenizer, model)
                daily = float(np.mean(scores))
            else:
                scores_list = [h.get("synthetic_score", 0.0) for h in items]
                daily = float(np.mean(scores_list))

            all_rows.append({
                "ticker": ticker,
                "date": date,
                "headline_count": len(items),
                "daily_sentiment": float(np.clip(daily, -1, 1)),
            })

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute rolling averages per ticker
    rolling_7d, rolling_30d, flags = [], [], []

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        series = df.loc[mask, "daily_sentiment"]

        r7 = series.rolling(7, min_periods=1).mean()
        r30 = series.rolling(30, min_periods=1).mean()

        rolling_7d.extend(r7.tolist())
        rolling_30d.extend(r30.tolist())
        # Bearish flag: 7-day avg dropped >0.15 below 30-day avg
        flags.extend((r7 < r30 - 0.15).tolist())

    df["rolling_7d"] = rolling_7d
    df["rolling_30d"] = rolling_30d
    df["sentiment_flag"] = flags

    # Save
    output_path = SENTIMENT_DIR / "sentiment_scores.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 6A — SENTIMENT SUMMARY")
    print("=" * 60)
    print(f"Total rows:       {len(df)}")
    print(f"Unique tickers:   {df['ticker'].nunique()}")
    print(f"Date range:       {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Sentiment flags:  {df['sentiment_flag'].sum()}")

    latest = df.groupby("ticker").last().reset_index()
    top_pos = latest.nlargest(5, "rolling_30d")[["ticker", "rolling_30d"]]
    top_neg = latest.nsmallest(5, "rolling_30d")[["ticker", "rolling_30d"]]

    print("\nTop 5 Most Positive (30d avg):")
    for _, row in top_pos.iterrows():
        print(f"  {row['ticker']:6s}: {row['rolling_30d']:+.4f}")
    print("\nTop 5 Most Negative (30d avg):")
    for _, row in top_neg.iterrows():
        print(f"  {row['ticker']:6s}: {row['rolling_30d']:+.4f}")

    return df


if __name__ == "__main__":
    build_sentiment_parquet()
