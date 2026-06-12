# FinRisk — Equity Risk Intelligence System

An end-to-end NLP pipeline that reads SEC 10-K filings, scores companies on financial health and news sentiment, and surfaces risk insights through a conversational retrieval interface. 

Built as a comprehensive PGDBA capstone project at IIM Calcutta.

---

## 📖 The Intuitive Guide: How FinRisk Works (Step-by-Step with Examples)

Imagine you want to invest in **Apple (AAPL)**. You want to know if they are facing a semiconductor shortage or an impending lawsuit. To find out, you'd normally have to read a 200-page legal document called a **10-K filing**, check the daily financial news, and run complex accounting formulas. 

**FinRisk does all of this for you automatically.** Here is the simple, step-by-step intuition behind how the pipeline works:

### 1. Getting the Documents (Ingestion)
**The Problem:** The SEC 10-K filing is a massive, boring PDF/HTML file.
**The Solution:** FinRisk automatically downloads these filings and acts like a smart highlighter. It ignores the fluff and only extracts three specific sections: *Risk Factors* (What could go wrong?), *Management's Discussion* (What does the CEO think?), and *Financial Statements* (The raw numbers).

### 2. Throwing out the Junk (Boilerplate Classification)
**The Problem:** Even in the "Risk Factors" section, lawyers add useless text like *"See Exhibit B on page 42 for signatures."*
**The Solution:** We train an AI (DistilBERT) to recognize what "lawyer junk" looks like. It scans the text and throws out the boilerplate, keeping only the actual risks (e.g., *"We rely on single-source suppliers in Asia"*).

### 3. Cutting into Bite-Sized Pieces (Section-Aware Chunking)
**The Problem:** AI models can't read a 50-page block of text all at once. They need small "chunks." But if we blindly chop the text every 500 words, we might cut a sentence in half!
**The Solution:** FinRisk uses **Section-Aware Chunking**. It looks for double-newlines to split by *paragraphs*. If a paragraph is too long, it splits by *sentences*. This ensures every piece of text the AI reads is a complete, coherent thought.

### 4. Building the Library Catalog (Indexing)
**The Problem:** When you ask a question, we can't read all 15,000 chunks of text to find the answer. It would take too long.
**The Solution:** We build two massive catalogs:
- **The Keyword Catalog (BM25):** Good for exact matches. If you search for *"TSMC"*, it instantly finds chunks containing that exact acronym.
- **The Concept Catalog (FAISS Dense Index):** Good for ideas. If you search for *"factory shutdown"*, it will find a chunk about *"manufacturing facility closure"* even though the exact words don't match.

### 5. The Smart Search (Hybrid Retrieval & Reranking)
**The Problem:** You ask, *"What are Apple's supply chain risks?"* Both catalogs return their top 20 guesses, but some might be irrelevant.
**The Solution:** We merge the 40 guesses together. Then, we use a very strict judge (a **Cross-Encoder AI**). The judge reads your question in one hand, and the text chunk in the other, and scores how perfectly they match. It throws away the bad ones and gives us the **Top 5 Absolute Best Chunks**.

### 6. Checking the News & Math (Sentiment & Z-Score)
**The Problem:** SEC filings only come out once a year. What if Apple's factory caught on fire *yesterday*? What if they are secretly running out of cash?
**The Solution:** 
- **News Sentiment:** We pull the last 30 days of news headlines. An AI reads them and scores them from -1 (Bad) to +1 (Good). *"Apple sales plummet"* gets a negative score.
- **Altman Z-Score:** We pull their balance sheet and run a classic bankruptcy math formula. If the score is low, they are in the "Distress Zone".

### 7. The Final Grade & Emerging Risks (Composite Score & YoY)
**The Solution:** We combine the SEC Filing Risk, the News Sentiment Risk, and the Math (Z-Score) Risk into a single grade out of 100. 
- **Year-over-Year (YoY) Check:** Did Apple add a brand new risk this year? We use AI to compare last year's filing to this year's filing. If the text is drastically different, we flag it as an "Emerging Risk" and highlight the exact new sentences.

### 8. The Chatbot (RAG - Retrieval-Augmented Generation)
**The Solution:** You ask the chatbot a question. We hand the **Top 5** chunks we found in Step 5 to a massive supercomputer brain (Groq Llama 3). We give it strict instructions: *"Answer the user's question using ONLY these 5 chunks. Do not make anything up. Cite your sources."* The result is a perfect, plain-English answer backed by real data.

---

## ⚙️ Architecture

```
SEC EDGAR (10-K filings)
         │
         ▼
  Section Extraction                News Headlines (NewsAPI)
  Item 1A — Risk Factors                 │
  Item 7  — MD&A                    FinBERT NLP (ProsusAI/finbert)
  Item 8  — Financial Statements         │
         │                     Sentiment Score (−1 to +1)
         ▼                              │
Section-Aware Chunking                  │
         │                         ┌───▼──────────┐
         ▼                         │   Composite  │
  BGE Embeddings ──► FAISS         │ Risk Profile │◄── Altman Z-Score
  Tokenized text ──► BM25          │              │    (yfinance data)
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

## 🔬 Pipeline Stages: Deep Dive

### Stage 1: Ingestion (`src/ingest.py`)
Downloads 10-K filings for 26 large-cap companies for the years 2022-2024 from SEC EDGAR using `sec-edgar-downloader`. Uses `BeautifulSoup` to strip HTML and extract three specific sections based on Regex patterns: Item 1A (Risk Factors), Item 7 (MD&A), and Item 8 (Financial Statements). The raw text is saved to `finrisk_sections.parquet`.

### Stage 2: Boilerplate Classification (`src/classify.py`)
Fine-tunes a `distilbert-base-uncased` model to classify text chunks into 4 classes: Risk Factors, MD&A, Financial Statements, and Boilerplate. The model is trained on a balanced dataset (capped at 200 samples/class) with a learning rate of `2e-5` over 3 epochs using `f1_macro` for early stopping. Inference is run over all sections to flag `is_boilerplate` rows.

### Stage 3: Chunking Strategies (`src/chunk.py`)
Implements two parallel chunking strategies:
1. **Fixed-Size Chunking**: Slices text into 512-token chunks with a 100-token overlap using the `tiktoken` `cl100k_base` tokenizer.
2. **Section-Aware Chunking (Proposed)**: Hierarchical splitting that respects natural boundaries. Splits by double newlines (paragraphs). If a paragraph > 512 tokens, it splits by sentences. This ensures coherent chunks that don't abruptly cut off semantic meaning.

### Stage 4: Indexing (`src/index.py`)
Builds 4 distinct indexes covering both chunking strategies:
- **FAISS (Dense)**: Embeds chunks using `BAAI/bge-small-en-v1.5` (384 dimensions). Uses `IndexFlatIP` (Inner Product) which computes exact Cosine Similarity since vectors are L2-normalized.
- **BM25 (Sparse)**: Tokenizes text and builds an exact-keyword index using `rank-bm25` (BM25Okapi algorithm).

### Stage 5: Hybrid Retrieval & Reranking (`src/retrieve.py`)
Implements a 3-step retrieval pipeline:
1. **Dense Search**: Retrieves Top-20 chunks from FAISS based on semantic embedding similarity.
2. **Sparse Search**: Retrieves Top-20 chunks from BM25 based on keyword exact-matches.
3. **Cross-Encoder Reranking**: Merges and deduplicates candidates, then scores each (Query, Document) pair through a single forward pass of `cross-encoder/ms-marco-MiniLM-L-6-v2`. This is highly accurate compared to bi-encoders. The Top-5 chunks are returned.

### Stage 6A: Sentiment Pipeline (`src/sentiment.py`)
Fetches the last 30 days of news headlines via NewsAPI (falls back to realistic synthetic data if API key is missing). Scores each headline using `ProsusAI/finbert`. Calculates a single score per headline: `P(Positive) - P(Negative)`, yielding a range of [-1.0, 1.0]. Calculates 7-day and 30-day rolling averages per company, and sets a bearish "sentiment flag" if the 7-day average drops more than 0.15 below the 30-day average.

### Stage 6B: Altman Z-Score (`src/zscore.py`)
Pulls balance sheet and financial data dynamically from `yfinance`. Calculates the classic Altman Z-Score bankruptcy prediction formula:
`Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5`
Where X1=Working Capital/Total Assets, X2=Retained Earnings/Total Assets, X3=EBIT/Total Assets, X4=Market Cap/Total Liabilities, X5=Revenue/Total Assets.
Classifies companies into: Safe (Z > 2.99), Grey Zone (1.81 ≤ Z ≤ 2.99), and Distress (Z < 1.81).

### Stage 7: Composite Risk Score & YoY Trends (`src/risk_score.py`)
- **Composite Score**: Min-max normalizes filing risk, inverts sentiment & Z-score metrics, and weights them: `0.40 * Filing + 0.35 * Sentiment + 0.25 * Z-Score`. The final 0-100 score dictates a Low (≤33), Elevated (≤66), or High (>66) label.
- **YoY Emerging Risks**: Computes cosine similarity between consecutive years of a company's Item 1A text using BGE embeddings. If similarity drops below the `0.85` threshold, it runs a sentence-level comparison. Sentences with a max similarity to the prior year of `< 0.72` are flagged as newly emerging risks.

### Stage 8: RAG & Evaluation (`src/rag.py`, `src/evaluate.py`)
- **RAG Generation**: Passes the user query and the Top-5 retrieved chunks into Groq's `llama-3.3-70b-versatile` API. The system prompt forces a strict JSON schema output: `summary`, `key_risks`, `confidence`, and `citations`. If the API key is missing, an extractive NLP fallback is used.
- **Evaluation**: Performs a Chunking Ablation Study to measure Hit@K, MRR, and NDCG. Runs RAGAS-style NLP evaluation scoring Faithfulness, Answer Relevance, and Context Precision.

---

## 📊 Research Results

### Chunking and Retrieval Ablation
| Method | Hit@5 | Hit@10 | MRR | NDCG@5 |
|--------|-------|--------|-----|--------|
| BM25 only | 0.200 | — | 0.117 | — |
| FAISS only | 0.250 | — | 0.102 | — |
| Hybrid fixed-size | 0.200 | 0.300 | 0.096 | 0.038 |
| **Hybrid section-aware** | **0.250** | **0.300** | **0.129** | **0.072** |

Section-aware chunking improves Mean Reciprocal Rank (MRR) by ~25% over fixed-size chunking. This strongly validates the hypothesis that respecting paragraph and sentence boundaries prevents the semantic destruction of context, leading to more coherent and accurate retrieval units for LLM reasoning.

### RAG Quality (RAGAS-style)
| Metric | Score |
|--------|-------|
| Faithfulness | 0.977 |
| Context Precision | 0.617 |
| Answer Relevance | 0.301 |

Faithfulness is remarkably high, ensuring the LLM does not hallucinate beyond the provided SEC filings. Answer Relevance is highly dependent on the active use of the Llama-3.3 model over the extractive fallback.

---

## 🚀 Setup Instructions

### 1. Clone and install dependencies
```bash
git clone https://github.com/pragathigudaru9/finrisk.git
cd finrisk/finrisk
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export GROQ_API_KEY="gsk_..."        # required for RAG responses
export NEWSAPI_KEY="..."             # optional — uses realistic synthetic data if missing
```

### 3. Run ingestion (one-time)
```bash
python -m src.ingest
```
*Note: This downloads 10-K filings for 26 companies (2022–2024) from SEC EDGAR directly to `data/processed/finrisk_sections.parquet`.*

### 4. Run the rest of the pipeline
Execute the orchestration script to sequentially run Stages 3-8:
```bash
python run_pipeline.py
```
Or run individual stages selectively:
```bash
python run_pipeline.py --stages 3 4        # chunk + index
python run_pipeline.py --stages 6a 6b 7    # sentiment + z-score + risk scoring
python run_pipeline.py --stages 8          # evaluation
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to view the 5-page dashboard.

---

## 💻 Streamlit Dashboard Pages (`app.py`)

1. **🏠 Company Overview**: Displays the company's composite Risk Score via Plotly gauges, a component breakdown, Z-Score zone, and a cross-sector leaderboard.
2. **🔍 Risk Explorer**: An interactive data-grid to search and browse the 15,000+ chunked filing excerpts by keyword and metadata filters.
3. **📰 Sentiment Trends**: Plots 30-day rolling FinBERT sentiment averages using Plotly, highlighting bearish trend crosses.
4. **🔄 Emerging Risks**: Displays a YoY cosine similarity heatmap and lists exact sentences flagged as new or removed risks.
5. **💬 Risk Assistant**: A conversational chatbot interface backed by the Hybrid RAG pipeline. Generates citations and confidence scores based on the retrieved context.

---

## ✅ Tests

```bash
python -m pytest tests/test_pipeline.py -v --tb=short
```
The comprehensive test suite covers 13 independent sections ensuring pipeline robustness: configuration validation, raw data integrity, chunk length constraints, FAISS/BM25 index validity, retrieval ranking (descending scores), Z-score mathematical logic checks, and API response typing.

---

## 📁 Project Structure

```
finrisk/
├── app.py                   # Streamlit dashboard (5 pages)
├── config.py                # Hyperparameters, API keys, paths, tickers
├── requirements.txt         # Core dependencies
├── run_pipeline.py          # Orchestration script for CLI
├── src/
│   ├── ingest.py            # Stage 1: SEC EDGAR download & regex parsing
│   ├── classify.py          # Stage 2: DistilBERT boilerplate fine-tuning
│   ├── chunk.py             # Stage 3: tiktoken & section-aware logic
│   ├── index.py             # Stage 4: FAISS FlatIP & BM25Okapi creation
│   ├── retrieve.py          # Stage 5: Hybrid retrieval + Cross-encoder rerank
│   ├── sentiment.py         # Stage 6A: FinBERT news sentiment
│   ├── zscore.py            # Stage 6B: yfinance Altman Z-Score calculation
│   ├── risk_score.py        # Stage 7: Composite scoring + YoY cosine similarity
│   ├── rag.py               # Stage 8A: Groq Llama-3 API + RAG system prompt
│   └── evaluate.py          # Stage 8B: MRR/NDCG Ablation + RAGAS evaluation
├── tests/
│   └── test_pipeline.py     # Pytest unit and integration tests
└── data/
    ├── processed/           # Parquet storage for pipeline state
    ├── eval/                # Evaluation JSON metrics
    ├── chunks/              # JSONL chunk dictionaries (ignored)
    ├── financials/          # Z-score parquets (ignored)
    ├── sentiment/           # Sentiment parquets (ignored)
    └── models/              # Saved PyTorch models, FAISS indexes, BM25 Pickles (ignored)
```

---

## 🛠️ Technologies Used

- **NLP & Retrieval**: FAISS, `rank-bm25`, `sentence-transformers` (BGE Embeddings), HuggingFace Transformers (DistilBERT), `tiktoken`.
- **LLM API**: Groq API (`llama-3.3-70b-versatile`).
- **Financial NLP**: FinBERT (`ProsusAI/finbert`).
- **Data Engineering**: `pandas`, `pyarrow` (Parquet), `numpy`.
- **Data Acquisition**: `yfinance`, NewsAPI, `sec-edgar-downloader`, `BeautifulSoup`.
- **Frontend**: Streamlit, Plotly.
- **Testing**: `pytest`.

---

## 🌎 Covered Universe (26 Equities)
- **Technology**: AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, META, INTC
- **Financials**: JPM, GS, BAC, MS, C
- **Healthcare**: JNJ, PFE, LLY, MRK, ABBV
- **Energy**: XOM, CVX, COP
- **Consumer**: WMT, PG, KO
- **Industrials**: BA, CAT, GE, DAL, AAL
