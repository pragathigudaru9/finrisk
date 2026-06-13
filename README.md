# FinRisk — Single-Company Risk Intelligence

Detects which risk categories are spiking this quarter, backed by exact SEC filing evidence.

Selects one company, downloads its latest 10-K and 10-Q filings, classifies every paragraph into 8 risk categories, flags where mentions have spiked quarter-over-quarter, and surfaces the exact filing sentences as evidence. Combines financial health signals (Altman Z-Score) and 30-day FinBERT news sentiment as supporting context. Answers follow-up questions via a conversational RAG interface.

Built as a PGDBA capstone project at IIM Calcutta.

---

## The Problem

A portfolio manager tracking 30 US equities cannot read every 10-K and 10-Q each quarter. Risk signals change quarter-to-quarter, but buried inside hundreds of pages of legal text. There is no tool that answers: **"What changed *this quarter*, and where is the proof?"**

FinRisk answers exactly that.

---

## What It Does

```
User selects ticker (e.g. NVDA)
           │
           ▼
SEC EDGAR — downloads:
  Latest 10-K (annual)
  Latest 10-Q (current quarter)
  Prior  10-Q  (previous quarter)
           │
           ▼
Section Extraction
  Item 1A — Risk Factors
  Item 7  — MD&A
  Item 8  — Financial Statements
           │
           ▼
Boilerplate Filter (DistilBERT heuristic)
           │
           ▼
Section-Aware Chunking (tiktoken)
           │
      ┌────┴────────────────────┐
      ▼                         ▼
Zero-Shot Risk Classification   FAISS + BM25 Hybrid Index
(facebook/bart-large-mnli)       │
8 categories per chunk          Cross-Encoder Reranker
      │                              │
Quarter-over-Quarter             RAG Answer
Spike Detection (>50%)          (Groq Llama-3.3-70b)
      │
Evidence Extraction
(top-3 sentences per flagged category)
      │
      └──────────────┐
                     ▼
        Supporting Financial Signals
        FinBERT Sentiment (30-day)
        Altman Z-Score (4 quarters)
                     │
                     ▼
        Streamlit Dashboard (4 tabs)
        Tab 1: Risk Changes Table
        Tab 2: Financial & Sentiment
        Tab 3: Evidence Chunks
        Tab 4: Ask FinRisk (RAG)
```

---

## How to Use

```bash
# 1. Install dependencies
cd finrisk/finrisk
pip install -r requirements.txt

# 2. Set API keys
export GROQ_API_KEY="gsk_..."      # required for RAG chatbot
export NEWSAPI_KEY="..."           # optional — uses synthetic data if missing

# 3. Pre-build indexes (one-time, ~20 min)
python run_pipeline.py --stages 3 4 6a 6b

# 4. Launch
streamlit run app.py
```

Then:
1. Select a company from the sidebar dropdown (e.g. `NVDA`)
2. The system downloads and processes its latest 10-K + 10-Q automatically
3. Navigate the 4 tabs:

| Tab | What it shows |
|-----|---------------|
| 📊 **Risk Changes** | 8-row table with previous vs current quarter mention counts. Flagged ⚠️ when mentions increase > 50%. Grouped bar chart. |
| 💰 **Financial & Sentiment** | Altman Z-Score trend (4 quarters) with Safe/Distress zone lines + 3 key ratios. FinBERT 30-day rolling sentiment chart. |
| 📄 **Evidence** | Exact sentences from the 10-K/10-Q for each flagged risk category, sorted by classifier confidence. |
| 💬 **Ask FinRisk** | Conversational RAG chatbot with suggested questions, cited source chunks, and confidence scoring. |

---

## How the Pipeline Works (Step by Step)

### Step 1: Download (Ingestion)
FinRisk downloads the company's latest **10-K** (annual report) and two most recent **10-Q** (quarterly reports) directly from SEC EDGAR. It then extracts three specific chapters using regex patterns: *Item 1A — Risk Factors*, *Item 7 — MD&A*, and *Item 8 — Financial Statements*. All other content (table of contents, signatures, exhibits) is ignored.

### Step 2: Clean the Junk (Boilerplate Filter)
Even within the extracted sections, lawyers insert useless filler like *"See Exhibit B on page 42."* A DistilBERT-based heuristic classifier detects and removes these passages, keeping only substantive risk language.

### Step 3: Chunking
The cleaned text is split into bite-sized chunks the AI can process. Two strategies are compared:
- **Fixed-size:** Strict 512-token windows with 100-token overlap (baseline).
- **Section-aware (default):** Splits on paragraph boundaries first, then sentence boundaries, ensuring no sentence is cut mid-thought.

### Step 4: Risk Category Classification
Each chunk is passed through **facebook/bart-large-mnli**, a large-vocabulary NLI model, with the 8 risk category labels as zero-shot classification targets. No fine-tuning required — it works purely from label descriptions.

The 8 categories: `Regulatory`, `Supply Chain`, `Demand`, `Liquidity`, `Cybersecurity`, `Legal`, `Credit`, `Geopolitical`.

### Step 5: Quarter-over-Quarter Spike Detection
Chunks are grouped by `period_label`: `current_quarter` (latest 10-Q) vs `previous_quarter` (prior 10-Q). For each category, the count of chunks is compared:
```
change_pct = (current - previous) / previous × 100
flagged = change_pct ≥ 50%
```
If `Cybersecurity` went from 5 chunks to 18 chunks, that is a **+260% spike** — flagged.

### Step 6: Evidence Extraction
For every flagged category, the Top-3 chunks sorted by zero-shot confidence score are extracted as evidence. These are the exact, verbatim sentences from the company's regulatory filing.

### Step 7: Supporting Signals
In parallel to the filing analysis:
- **FinBERT Sentiment:** 30 days of news headlines are scored using `ProsusAI/finbert`. Each headline gets `P(Positive) - P(Negative)` in `[-1, +1]`. 7-day and 30-day rolling averages are plotted.
- **Altman Z-Score:** Balance sheet ratios from `yfinance` are fed into the classic bankruptcy prediction formula `Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5`. Companies land in **Safe** (Z > 2.99), **Grey Zone**, or **Distress** (Z < 1.81).

### Step 8: RAG Chatbot
You type any natural language question. The system runs hybrid retrieval (FAISS dense + BM25 sparse) to find the Top-20 candidate chunks, reranks them with a `cross-encoder/ms-marco-MiniLM-L-6-v2`, and passes the Top-5 to Groq Llama-3.3-70b with a strict instruction: *"Answer using only these chunks. Cite your sources."*

---

## Research Results

### Chunking Strategy Ablation

| Method | Hit@5 | Hit@10 | MRR | NDCG@5 |
|--------|-------|--------|-----|--------|
| BM25 only | 0.200 | — | 0.117 | — |
| FAISS only | 0.250 | — | 0.102 | — |
| Hybrid fixed-size | 0.200 | 0.300 | 0.096 | 0.038 |
| **Hybrid section-aware** | **0.250** | **0.300** | **0.129** | **0.072** |

Section-aware chunking improves MRR by **+25%** over fixed-size. Respecting paragraph and sentence boundaries preserves semantic coherence, leading to more relevant retrieval.

### RAG Quality (RAGAS-style)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.977 |
| Context Precision | 0.617 |
| Answer Relevance | 0.301 |

Faithfulness of 0.977 means the LLM almost never introduces information beyond the retrieved filing passages.

### Risk Category Classification
Zero-shot `facebook/bart-large-mnli` evaluated against 25 hand-labeled SEC filing chunks across all 8 risk categories. Target: ≥ 75% accuracy. Run `python run_pipeline.py --stages 8` to reproduce.

---

## Running Evaluation

```bash
# Chunking ablation + RAGAS + risk extraction accuracy
python run_pipeline.py --stages 8

# Individual stages for a specific company
python run_pipeline.py --ticker NVDA --stages 1 3 4
```

---

## Project Structure

```
finrisk/
├── app.py                   # Streamlit dashboard (4 tabs)
├── config.py                # Tickers, risk categories, model names, hyperparameters
├── requirements.txt
├── run_pipeline.py          # CLI: --ticker + --stages
├── src/
│   ├── ingest.py            # SEC EDGAR download + section parsing
│   │                        # download_company_latest() for on-demand single-ticker
│   ├── classify.py          # DistilBERT boilerplate heuristic
│   ├── chunk.py             # Fixed-size + section-aware chunking (tiktoken)
│   ├── index.py             # FAISS IndexFlatIP + BM25Okapi
│   ├── retrieve.py          # Hybrid retrieval + cross-encoder reranking
│   ├── extract_risks.py     # Zero-shot risk category classification (BART)
│   │                        # count_by_quarter(), extract_evidence()
│   ├── risk_score.py        # get_risk_summary() — single-company risk object
│   ├── sentiment.py         # FinBERT sentiment + get_sentiment_summary()
│   ├── zscore.py            # Altman Z-Score + get_zscore_history()
│   ├── rag.py               # Groq Llama-3.3-70b RAG
│   └── evaluate.py          # Ablation + RAGAS + evaluate_risk_extraction()
├── tests/
│   └── test_pipeline.py
└── data/
    ├── eval/
    │   ├── labeled_risk_chunks.json   # 25 hand-labeled examples
    │   ├── ablation_results.json
    │   ├── ragas_results.json
    │   └── risk_extraction_results.json
    ├── processed/            # finrisk_sections.parquet, risk_scores.parquet
    ├── chunks/               # JSONL chunk files (gitignored — large)
    ├── financials/           # zscore.parquet (gitignored)
    ├── sentiment/            # sentiment_scores.parquet (gitignored)
    └── models/               # FAISS indexes, BM25 pickles (gitignored — large)
```

---

## Technologies

- **NLP / Retrieval:** `sentence-transformers` (BAAI/bge-small-en-v1.5), FAISS, BM25Okapi, `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Risk Classification:** `facebook/bart-large-mnli` (zero-shot NLI)
- **Sentiment:** `ProsusAI/finbert` (financial domain FinBERT)
- **Financial Data:** `yfinance` (Altman Z-Score), `sec-edgar-downloader`, `BeautifulSoup`
- **LLM:** Groq API (`llama-3.3-70b-versatile`)
- **Frontend:** Streamlit, Plotly
- **ML Infrastructure:** HuggingFace Transformers, PyTorch, `tiktoken`, `pandas`

---

## Resume Line

*"Built a single-company equity risk intelligence system detecting emerging risk categories from SEC 10-K and 10-Q filings using zero-shot classification (facebook/bart-large-mnli), with quarter-over-quarter spike detection. Combined FinBERT news sentiment and Altman Z-Score as supporting financial signals. Benchmarked section-aware vs fixed-size chunking (MRR +25%), evaluated with RAGAS (faithfulness 0.977). Deployed as a focused 4-tab Streamlit dashboard with evidence-backed risk insights and a conversational RAG interface powered by Groq Llama-3.3-70b."*

---

## Companies Available (29 Tickers)

**Technology:** AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META, INTC  
**Financials:** JPM, GS, BAC, MS, C  
**Healthcare:** JNJ, PFE, LLY, MRK, ABBV  
**Energy:** XOM, CVX, COP  
**Consumer:** WMT, PG, KO  
**Industrials:** BA, CAT, GE, DAL, AAL
