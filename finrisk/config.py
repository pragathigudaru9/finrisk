"""
FinRisk — Central Configuration
All paths, API keys, company lists, model names, and hyperparameters.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_filings"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
SENTIMENT_DIR = DATA_DIR / "sentiment"
FINANCIALS_DIR = DATA_DIR / "financials"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = PROJECT_ROOT / "models"

# Create all directories on import
for d in [
    RAW_DIR,
    PROCESSED_DIR,
    CHUNKS_DIR / "fixed",
    CHUNKS_DIR / "section_aware",
    SENTIMENT_DIR,
    FINANCIALS_DIR,
    EVAL_DIR,
    MODELS_DIR / "section_classifier",
    MODELS_DIR / "faiss_fixed",
    MODELS_DIR / "faiss_section_aware",
]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# SEC EDGAR credentials (required by sec-edgar-downloader v5.x)
SEC_COMPANY_NAME = os.getenv("SEC_COMPANY_NAME", "FinRiskResearch")
SEC_EMAIL = os.getenv("SEC_EMAIL", "finrisk@research.edu")

# ──────────────────────────────────────────────────────────────
# Available tickers for single-company analysis
# ──────────────────────────────────────────────────────────────
AVAILABLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META", "INTC",
    "JPM", "GS", "BAC", "MS", "C",
    "JNJ", "PFE", "LLY", "MRK", "ABBV",
    "XOM", "CVX", "COP",
    "WMT", "PG", "KO",
    "BA", "CAT", "GE", "DAL", "AAL",
]

# Legacy alias kept for modules that reference ALL_TICKERS
ALL_TICKERS = AVAILABLE_TICKERS

# Ticker → full company name
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet (Google)",
    "META": "Meta Platforms", "NVDA": "NVIDIA", "TSLA": "Tesla",
    "AMD": "AMD", "INTC": "Intel",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup",
    "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "MRK": "Merck",
    "ABBV": "AbbVie", "LLY": "Eli Lilly",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "BA": "Boeing", "CAT": "Caterpillar", "GE": "General Electric",
    "DAL": "Delta Air Lines", "AAL": "American Airlines",
}

# Sector mapping
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "TSLA": "Consumer Discretionary",
    "AMD": "Technology", "META": "Technology", "INTC": "Technology",
    "JPM": "Financials", "GS": "Financials", "BAC": "Financials",
    "MS": "Financials", "C": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "LLY": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "DAL": "Industrials", "AAL": "Industrials",
}

# ──────────────────────────────────────────────────────────────
# Zero-shot risk categories (for extract_risks.py)
# ──────────────────────────────────────────────────────────────
RISK_CATEGORIES = [
    "Regulatory", "Supply Chain", "Demand", "Liquidity",
    "Cybersecurity", "Legal", "Credit", "Geopolitical",
]

# Flag a category when quarter-over-quarter mention count increases > this %
RISK_SPIKE_THRESHOLD = 0.50  # 50%

# ──────────────────────────────────────────────────────────────
# Model names
# ──────────────────────────────────────────────────────────────
BGE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"         # 130MB — fast, space-efficient
BGE_RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 82MB — fast cross-encoder
FINBERT_MODEL = "ProsusAI/finbert"
SECTION_CLASSIFIER_MODEL = "distilbert-base-uncased"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

# ──────────────────────────────────────────────────────────────
# Chunking config
# ──────────────────────────────────────────────────────────────
FIXED_CHUNK_SIZE = 512       # tokens
FIXED_CHUNK_OVERLAP = 100    # tokens

# ──────────────────────────────────────────────────────────────
# Retrieval config
# ──────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20         # candidates from hybrid search (each leg)
TOP_K_RERANKED = 5           # final chunks after reranking
