# FinRisk — Equity Risk Intelligence System

An end-to-end NLP pipeline that reads SEC 10-K filings, scores companies on
financial health and news sentiment, and surfaces risk insights through a
conversational retrieval interface.

Built as part of a PGDBA capstone project at IIM Calcutta.

---

## Motivation

Portfolio managers tracking 15–30 US equities need to stay on top of risk
disclosures buried across hundreds of pages of regulatory filings, financial
statements, and daily news. Reading every 10-K manually is not feasible.

This project automates that process: it downloads filings, extracts the
three most risk-relevant sections, chunks and indexes them, runs sentiment
analysis and Altman Z-Score computation in parallel, and then combines
everything into a single composite risk score per company. A five-page
Streamlit dashboard and a conversational retrieval interface make the results
accessible without writing any queries.

---

## Architecture

```
SEC EDGAR (10-K filings)
         │
         ▼
  Section Extraction                News Headlines
  Item 1A — Risk Factors                 │
  Item 7  — MD&A                    FinBERT NLP
  Item 8  — Financial Statements         │
         │                     Sentiment Score (−1 to +1)
         ▼                              │
Section-Aware Chunking                  │
         │                         ┌───▼──────────┐
         ▼                         │   Composite   │
  BGE Embeddings ──► FAISS         │  Risk Profile │◄── Altman Z-Score
  Tokenized text ──► BM25          │               │    (yfinance data)
                                   └───────┬───────┘
  Hybrid Retrieval (FAISS + BM25)          │
         │                                 │
  Cross-Encoder Reranker                   │
         │                                 │
   Top-5 Chunks                            │
         │                                 │
  Groq Llama-3.3-70b                       │
  (or extractive fallback)                 │
         │                                 ▼
   Risk Explanation ──────────► Streamlit Dashboard
```

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `src/ingest.py` | Download 10-K filings from SEC EDGAR and parse Risk Factors, MD&A, Financial Statements using BeautifulSoup + regex |
| 2 | `src/classify.py` | Fine-tune DistilBERT to detect and filter boilerplate text from parsed sections |
| 3 | `src/chunk.py` | Split sections using two strategies: fixed 512-token windows (with overlap) and paragraph-aware section chunking |
| 4 | `src/index.py` | Build FAISS dense index (BGE embeddings) and BM25 sparse index for both chunking strategies |
| 5 | `src/retrieve.py` | Hybrid retrieval: merge FAISS + BM25 candidates, rerank with a cross-encoder, return top-5 |
| 6A | `src/sentiment.py` | Fetch news headlines via NewsAPI, score with FinBERT, compute 7-day and 30-day rolling averages |
| 6B | `src/zscore.py` | Pull balance sheet data from yfinance and compute Altman Z-Score for each company |
| 7 | `src/risk_score.py` | Combine filing risk, sentiment risk, and Z-Score risk into a weighted composite score (0–100); detect YoY risk language changes using cosine similarity |
| 8A | `src/rag.py` | Answer natural language queries using the retrieved chunks and Groq Llama-3.3-70b |
| 8B | `src/evaluate.py` | Chunking ablation study (Hit@5, Hit@10, MRR, NDCG) and RAGAS-style quality evaluation |
| 9 | `app.py` | Five-page Streamlit dashboard |

---

## Research Results

### Chunking and Retrieval Ablation

| Method | Hit@5 | Hit@10 | MRR | NDCG@5 |
|--------|-------|--------|-----|--------|
| BM25 only | 0.200 | — | 0.117 | — |
| FAISS only | 0.250 | — | 0.102 | — |
| Hybrid fixed-size | 0.200 | 0.300 | 0.096 | 0.038 |
| **Hybrid section-aware** | **0.250** | **0.300** | **0.129** | **0.072** |

Section-aware chunking improves MRR by ~25% over fixed-size chunking, which
validates the hypothesis that respecting paragraph boundaries leads to more
coherent retrieval units.

### RAG Quality (RAGAS-style)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.977 |
| Answer Relevance | 0.301 |
| Context Precision | 0.617 |

Faithfulness is high because answers are constrained to text that appears in
the retrieved chunks. Answer relevance improves significantly when the Groq
LLM is active versus the extractive fallback.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/pragathigudaru9/finrisk.git
cd finrisk/finrisk
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export GROQ_API_KEY="gsk_..."        # required for RAG responses
export NEWSAPI_KEY="..."             # optional — uses synthetic data if missing
```

### 3. Run ingestion (one-time)

```bash
python -m src.ingest
```

This downloads 10-K filings for 26 companies (2022–2024) from SEC EDGAR
and writes `data/processed/finrisk_sections.parquet`.

### 4. Run the rest of the pipeline

```bash
python run_pipeline.py
```

Or run individual stages:

```bash
python run_pipeline.py --stages 3 4        # chunk + index
python run_pipeline.py --stages 6a 6b 7    # sentiment + z-score + risk scoring
python run_pipeline.py --stages 8          # evaluation
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501`

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Company Overview | Risk gauge, Z-Score, component breakdown, sector leaderboard |
| 🔍 Risk Explorer | Search and browse 15,000+ filing excerpts by keyword and metadata filter |
| 📰 Sentiment Trends | 30-day rolling FinBERT sentiment charts with bearish flag detection |
| 🔄 Emerging Risks | YoY cosine similarity heatmap; sentence-level new/removed risk detection |
| 💬 Risk Assistant | Conversational interface backed by hybrid RAG with source citations |

---

## Key Technical Concepts

**Hybrid retrieval** — Combining dense (FAISS) and sparse (BM25) search
captures both semantic similarity and exact keyword overlap. Dense search
finds conceptually related passages; BM25 catches specific names, acronyms,
and regulatory terms.

**Cross-encoder reranking** — Standard bi-encoder embeddings process the
query and document independently, which trades speed for some accuracy. A
cross-encoder sees the query and candidate together in a single forward pass
through the Transformer, giving a much more precise relevance score. We apply
it only to the top-40 candidates from the first-stage retrieval.

**FinBERT sentiment** — BERT fine-tuned on financial news text. Outputs
probability scores for positive, negative, and neutral classes. We compute
`p_positive − p_negative` as a single score in [−1, +1] and track 7-day and
30-day rolling averages per company.

**Altman Z-Score** — Classic bankruptcy prediction formula:

```
Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5

X1 = Working Capital / Total Assets
X2 = Retained Earnings / Total Assets
X3 = EBIT / Total Assets
X4 = Market Cap / Total Liabilities
X5 = Revenue / Total Assets
```

Z > 2.99 → Safe   |   1.81 ≤ Z ≤ 2.99 → Grey Zone   |   Z < 1.81 → Distress

**YoY emerging risk detection** — The Risk Factors section from consecutive
filing years is embedded and compared via cosine similarity. If the similarity
drops below 0.85, the year-pair is flagged. Sentence-level comparison then
identifies exactly which risk sentences are new or have been removed.

---

## Tests

```bash
pip install pytest
python -m pytest finrisk/tests/ -v --tb=short
```

The test suite has 13 sections covering configuration, raw data integrity,
chunking correctness, index validity, retrieval quality, risk scores,
sentiment data, Z-scores, YoY trends, RAG output format, evaluation data,
chunking unit tests, and Z-score formula logic.

---

## Project Structure

```
RAG/
├── .gitignore
├── README.md
├── data.ipynb                   # early-stage data exploration
└── finrisk/
    ├── app.py                   # Streamlit dashboard (5 pages)
    ├── config.py                # central config — paths, tickers, hyperparameters
    ├── requirements.txt
    ├── run_pipeline.py          # orchestrate stages 3–8 from CLI
    ├── src/
    │   ├── ingest.py            # Stage 1: SEC EDGAR download + section parsing
    │   ├── classify.py          # Stage 2: DistilBERT boilerplate classifier
    │   ├── chunk.py             # Stage 3: fixed-size + section-aware chunking
    │   ├── index.py             # Stage 4: FAISS + BM25 index building
    │   ├── retrieve.py          # Stage 5: hybrid retrieval + cross-encoder rerank
    │   ├── sentiment.py         # Stage 6A: FinBERT news sentiment
    │   ├── zscore.py            # Stage 6B: Altman Z-Score
    │   ├── risk_score.py        # Stage 7: composite scoring + YoY trends
    │   ├── rag.py               # Stage 8A: RAG with Groq Llama-3.3-70b
    │   └── evaluate.py          # Stage 8B: ablation + RAGAS evaluation
    ├── tests/
    │   └── test_pipeline.py
    └── data/
        ├── processed/           # finrisk_sections.parquet, risk_scores.parquet
        │                        # yoy_trends.parquet  (git-tracked)
        ├── eval/                # qa_pairs.json, ablation_results.json,
        │                        # ragas_results.json  (git-tracked)
        ├── chunks/              # .jsonl chunk files (gitignored — large)
        ├── financials/          # zscore.parquet (gitignored)
        ├── sentiment/           # sentiment_scores.parquet (gitignored)
        └── models/              # FAISS indexes, BM25 pickles (gitignored — large)
```

---

## Technologies

Python · FAISS · BM25 (`rank-bm25`) · sentence-transformers (BGE) ·
Groq API (Llama-3.3-70b) · FinBERT (ProsusAI) · HuggingFace Transformers ·
yfinance · NewsAPI · Streamlit · Plotly · pandas · tiktoken · PyTorch ·
BeautifulSoup · sec-edgar-downloader

---

## Companies Covered

26 US large-cap equities across six sectors, with three years of 10-K
filings each (2022–2024 filing years):

**Technology** — AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META  
**Financials** — JPM, GS, BAC, MS  
**Healthcare** — JNJ, PFE, LLY, MRK, ABBV  
**Energy** — XOM, CVX, COP  
**Consumer** — WMT, PG, KO  
**Industrials** — BA, CAT, DAL, AAL
