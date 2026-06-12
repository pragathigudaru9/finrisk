"""
FinRisk — Comprehensive Test Suite

Tests cover:
  - Configuration & paths
  - Data pipeline (chunks, indexes, risk scores, sentiment, Z-scores)
  - Retrieval (hybrid, FAISS, BM25)
  - RAG (Groq + extractive fallback)
  - Evaluation metrics
  - App data loading
"""

import json
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import (
    PROCESSED_DIR, CHUNKS_DIR, MODELS_DIR, FINANCIALS_DIR,
    SENTIMENT_DIR, EVAL_DIR, GROQ_API_KEY, COMPANIES, ALL_TICKERS,
)


# ════════════════════════════════════════════════════════════════
# SECTION 1: Configuration
# ════════════════════════════════════════════════════════════════

class TestConfig:
    def test_all_directories_exist(self):
        for d in [PROCESSED_DIR, CHUNKS_DIR, MODELS_DIR, FINANCIALS_DIR, SENTIMENT_DIR, EVAL_DIR]:
            assert d.exists(), f"Directory missing: {d}"

    def test_companies_list(self):
        assert len(COMPANIES) >= 10, "Should have at least 10 companies"
        assert "AAPL" in COMPANIES
        assert "MSFT" in COMPANIES
        assert "JPM" in COMPANIES

    def test_all_tickers_list(self):
        assert len(ALL_TICKERS) >= 20, "ALL_TICKERS should have 20+ tickers"

    def test_groq_api_key_set(self):
        assert GROQ_API_KEY, "GROQ_API_KEY must be set"
        assert GROQ_API_KEY.startswith("gsk_"), "Groq API key should start with gsk_"

    def test_model_names(self):
        assert "bge" in config.BGE_EMBEDDING_MODEL.lower()
        # Reranker may be BGE or MiniLM cross-encoder (both valid)
        assert "bge" in config.BGE_RERANKER_MODEL.lower() or "cross-encoder" in config.BGE_RERANKER_MODEL.lower()
        assert "finbert" in config.FINBERT_MODEL.lower()

    def test_weights_sum_to_one(self):
        total = config.WEIGHT_FILING + config.WEIGHT_SENTIMENT + config.WEIGHT_ZSCORE
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


# ════════════════════════════════════════════════════════════════
# SECTION 2: Raw Data
# ════════════════════════════════════════════════════════════════

class TestRawData:
    @pytest.fixture(autouse=True)
    def load(self):
        self.sections = pd.read_parquet(PROCESSED_DIR / "finrisk_sections.parquet")

    def test_sections_not_empty(self):
        assert len(self.sections) > 100, "Should have 100+ section rows"

    def test_sections_columns(self):
        required = {"ticker", "year", "section_type", "raw_text"}
        assert required.issubset(self.sections.columns)

    def test_sections_tickers(self):
        tickers = self.sections["ticker"].unique()
        assert len(tickers) >= 20, f"Expected 20+ tickers, got {len(tickers)}"

    def test_sections_years(self):
        years = self.sections["year"].unique()
        assert len(years) >= 2, "Should have at least 2 years"
        assert all(y >= 2023 for y in years)

    def test_sections_section_types(self):
        types = set(self.sections["section_type"].unique())
        assert "Risk Factors" in types

    def test_sections_text_not_empty(self):
        empty = self.sections[self.sections["raw_text"].str.len() < 100]
        assert len(empty) == 0, f"{len(empty)} sections have very short text"


# ════════════════════════════════════════════════════════════════
# SECTION 3: Chunking
# ════════════════════════════════════════════════════════════════

class TestChunking:
    @pytest.fixture(autouse=True)
    def load(self):
        self.fixed_path = CHUNKS_DIR / "fixed" / "chunks.jsonl"
        self.section_path = CHUNKS_DIR / "section_aware" / "chunks.jsonl"
        self.fixed = self._load(self.fixed_path)
        self.section = self._load(self.section_path)

    def _load(self, path):
        chunks = []
        with open(path) as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def test_chunk_files_exist(self):
        assert self.fixed_path.exists()
        assert self.section_path.exists()

    def test_fixed_chunks_count(self):
        assert len(self.fixed) > 5000, f"Expected 5000+ fixed chunks, got {len(self.fixed)}"

    def test_section_chunks_count(self):
        assert len(self.section) > 5000, f"Expected 5000+ section chunks, got {len(self.section)}"

    def test_fixed_avg_tokens(self):
        avg = sum(c["token_count"] for c in self.fixed) / len(self.fixed)
        assert 400 <= avg <= 515, f"Fixed avg tokens {avg:.0f} outside [400, 515]"

    def test_required_chunk_fields(self):
        required = {"chunk_id", "ticker", "year", "section_type", "text", "token_count"}
        for chunk in self.fixed[:10]:
            assert required.issubset(chunk.keys()), f"Missing fields in chunk: {chunk.keys()}"

    def test_chunk_ids_unique(self):
        ids = [c["chunk_id"] for c in self.fixed]
        assert len(ids) == len(set(ids)), "Fixed chunk IDs are not unique"

    def test_text_not_empty(self):
        empty = [c for c in self.fixed if len(c["text"].strip()) < 10]
        assert len(empty) == 0, f"{len(empty)} fixed chunks have near-empty text"

    def test_same_tickers_both_strategies(self):
        fixed_t = set(c["ticker"] for c in self.fixed)
        sect_t = set(c["ticker"] for c in self.section)
        assert fixed_t == sect_t, "Ticker sets differ between strategies"


# ════════════════════════════════════════════════════════════════
# SECTION 4: Indexes
# ════════════════════════════════════════════════════════════════

class TestIndexes:
    def test_all_index_files_exist(self):
        files = [
            MODELS_DIR / "faiss_fixed" / "index.faiss",
            MODELS_DIR / "faiss_section_aware" / "index.faiss",
            MODELS_DIR / "bm25_fixed.pkl",
            MODELS_DIR / "bm25_section_aware.pkl",
        ]
        for f in files:
            assert f.exists(), f"Missing index file: {f}"

    def test_faiss_index_non_empty(self):
        import faiss
        for strategy in ["fixed", "section_aware"]:
            index = faiss.read_index(str(MODELS_DIR / f"faiss_{strategy}" / "index.faiss"))
            assert index.ntotal > 1000, f"{strategy} FAISS index has only {index.ntotal} vectors"

    def test_faiss_chunks_meta_exists(self):
        for strategy in ["fixed", "section_aware"]:
            meta_p = MODELS_DIR / f"faiss_{strategy}" / "chunks_meta.json"
            texts_p = MODELS_DIR / f"faiss_{strategy}" / "chunks_texts.json"
            assert meta_p.exists()
            assert texts_p.exists()

    def test_bm25_loads(self):
        import pickle
        for strategy in ["fixed", "section_aware"]:
            with open(MODELS_DIR / f"bm25_{strategy}.pkl", "rb") as f:
                data = pickle.load(f)
            assert "bm25" in data
            assert "mapping" in data
            assert len(data["mapping"]) > 1000

    def test_faiss_dim_correct(self):
        import faiss
        index = faiss.read_index(str(MODELS_DIR / "faiss_section_aware" / "index.faiss"))
        assert index.d in (384, 768), f"Expected 384 (bge-small) or 768 (bge-base) dims, got {index.d}"


# ════════════════════════════════════════════════════════════════
# SECTION 5: Retrieval
# ════════════════════════════════════════════════════════════════

class TestRetrieval:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self, tmp_path_factory):
        from src.retrieve import HybridRetriever
        self.__class__.retriever = HybridRetriever("section_aware")

    def test_retriever_initialized(self):
        assert self.retriever is not None

    def test_returns_top_5(self):
        results = self.retriever.retrieve("What are Apple's risks?", top_k=5)
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    def test_returns_top_10(self):
        results = self.retriever.retrieve("cybersecurity threats", top_k=10)
        assert len(results) == 10

    def test_all_required_fields(self):
        results = self.retriever.retrieve("supply chain risks", top_k=5)
        required = {"chunk_id", "ticker", "year", "section_type", "text",
                    "faiss_score", "bm25_score", "reranker_score", "final_rank"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing fields: {required - r.keys()}"

    def test_reranker_scores_descending(self):
        results = self.retriever.retrieve("interest rate risk banks", top_k=5)
        scores = [r["reranker_score"] for r in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
            f"Scores not descending: {scores}"

    def test_jpm_query_relevance(self):
        results = self.retriever.retrieve("What credit risks did JPMorgan disclose?", top_k=5)
        jpm_matches = sum(1 for r in results
                          if r["ticker"] == "JPM" or "jpmorgan" in r["text"].lower())
        assert jpm_matches >= 2, f"Expected ≥2 JPM results, got {jpm_matches}"

    def test_text_is_not_empty(self):
        results = self.retriever.retrieve("semiconductor supply chain NVIDIA", top_k=5)
        for r in results:
            assert len(r["text"]) > 50, "Retrieved chunk text is too short"

    def test_final_rank_sequential(self):
        results = self.retriever.retrieve("climate risk oil companies", top_k=5)
        ranks = [r["final_rank"] for r in results]
        assert ranks == list(range(1, len(ranks) + 1)), f"Ranks not sequential: {ranks}"


# ════════════════════════════════════════════════════════════════
# SECTION 6: Risk Scores
# ════════════════════════════════════════════════════════════════

class TestRiskScores:
    @pytest.fixture(autouse=True)
    def load(self):
        self.scores = pd.read_parquet(PROCESSED_DIR / "risk_scores.parquet")

    def test_risk_scores_loaded(self):
        assert len(self.scores) >= 20, f"Expected 20+ companies, got {len(self.scores)}"

    def test_all_scores_in_range(self):
        assert self.scores["risk_score"].between(0, 100).all()

    def test_risk_labels_valid(self):
        valid = {"Low", "Elevated", "High"}
        assert set(self.scores["risk_label"].unique()).issubset(valid)

    def test_labels_match_scores(self):
        for _, row in self.scores.iterrows():
            expected = "Low" if row["risk_score"] <= 33 else \
                       ("Elevated" if row["risk_score"] <= 66 else "High")
            assert row["risk_label"] == expected, \
                f"{row['ticker']} label {row['risk_label']} != expected {expected}"

    def test_all_components_present(self):
        for col in ["filing_component", "sentiment_component", "zscore_component"]:
            assert col in self.scores.columns
            assert self.scores[col].between(0, 101).all()

    def test_sector_field_present(self):
        assert "sector" in self.scores.columns
        assert self.scores["sector"].notna().all()

    def test_diverse_risk_labels(self):
        # Should not be all the same label
        n_unique = self.scores["risk_label"].nunique()
        assert n_unique >= 2, f"Expected diverse risk labels, got only: {self.scores['risk_label'].unique()}"


# ════════════════════════════════════════════════════════════════
# SECTION 7: Sentiment
# ════════════════════════════════════════════════════════════════

class TestSentiment:
    @pytest.fixture(autouse=True)
    def load(self):
        self.df = pd.read_parquet(SENTIMENT_DIR / "sentiment_scores.parquet")

    def test_sentiment_loaded(self):
        assert len(self.df) > 100

    def test_columns_present(self):
        required = {"ticker", "date", "daily_sentiment", "rolling_7d", "rolling_30d", "sentiment_flag"}
        assert required.issubset(self.df.columns)

    def test_sentiment_range(self):
        assert self.df["daily_sentiment"].between(-1.001, 1.001).all()
        assert self.df["rolling_30d"].between(-1.001, 1.001).all()

    def test_multiple_tickers(self):
        assert self.df["ticker"].nunique() >= 20

    def test_date_column_is_datetime(self):
        assert pd.api.types.is_datetime64_any_dtype(self.df["date"])


# ════════════════════════════════════════════════════════════════
# SECTION 8: Z-Score
# ════════════════════════════════════════════════════════════════

class TestZScore:
    @pytest.fixture(autouse=True)
    def load(self):
        self.df = pd.read_parquet(FINANCIALS_DIR / "zscore.parquet")

    def test_zscore_loaded(self):
        assert len(self.df) >= 20

    def test_zone_column(self):
        valid_zones = {"Safe", "Grey Zone", "Distress"}
        assert set(self.df["zone"].unique()).issubset(valid_zones)

    def test_zones_match_scores(self):
        for _, row in self.df.iterrows():
            z = row["z_score"]
            expected = "Safe" if z > 2.99 else ("Grey Zone" if z >= 1.81 else "Distress")
            assert row["zone"] == expected, f"{row['ticker']} zone mismatch: z={z:.2f}"

    def test_diverse_zones(self):
        n_zones = self.df["zone"].nunique()
        assert n_zones >= 2, "Expected at least 2 different Z-score zones"


# ════════════════════════════════════════════════════════════════
# SECTION 9: YoY Trends
# ════════════════════════════════════════════════════════════════

class TestYoYTrends:
    @pytest.fixture(autouse=True)
    def load(self):
        p = PROCESSED_DIR / "yoy_trends.parquet"
        self.df = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def test_trends_not_empty(self):
        assert not self.df.empty, "YoY trends parquet is empty"

    def test_required_columns(self):
        required = {"ticker", "year_pair", "cosine_similarity", "change_type"}
        assert required.issubset(self.df.columns)

    def test_similarity_range(self):
        assert self.df["cosine_similarity"].between(-1.001, 1.001).all()

    def test_change_types_valid(self):
        valid = {"Stable", "Risk Language Changed"}
        assert set(self.df["change_type"].unique()).issubset(valid)

    def test_some_changes_detected(self):
        changed = self.df[self.df["change_type"] == "Risk Language Changed"]
        assert len(changed) >= 3, f"Expected ≥3 year-pairs flagged, got {len(changed)}"


# ════════════════════════════════════════════════════════════════
# SECTION 10: RAG Pipeline
# ════════════════════════════════════════════════════════════════

class TestRAG:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        from src.rag import FinRiskRAG
        self.__class__.rag = FinRiskRAG()

    def test_rag_initialized(self):
        assert self.rag is not None

    def test_query_returns_dict(self):
        result = self.rag.query("What are Apple's risks?")
        assert isinstance(result, dict)

    def test_query_has_answer(self):
        result = self.rag.query("What cybersecurity risks does Microsoft face?")
        assert "answer" in result
        assert isinstance(result["answer"], dict)

    def test_query_has_chunks(self):
        result = self.rag.query("How does Tesla describe supply chain risks?")
        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) > 0

    def test_answer_has_summary(self):
        result = self.rag.query("What credit risks did JPMorgan disclose?")
        answer = result["answer"]
        assert "summary" in answer
        assert len(answer["summary"]) > 20

    def test_answer_has_confidence(self):
        result = self.rag.query("What are NVIDIA semiconductor risks?")
        answer = result["answer"]
        assert answer.get("confidence") in {"high", "medium", "low"}

    def test_extractive_fallback(self):
        from src.rag import extractive_fallback
        chunks = [{"text": "Apple faces significant supply chain risks.", "ticker": "AAPL",
                   "year": 2024, "section_type": "Risk Factors"}]
        result = extractive_fallback("supply chain", chunks)
        assert "summary" in result
        assert "key_risks" in result
        assert "citations" in result

    def test_groq_call(self):
        from src.rag import call_groq, RAG_SYSTEM_PROMPT
        response = call_groq(RAG_SYSTEM_PROMPT, "Say hello in JSON with key 'summary'.")
        # API key may be expired in CI — accept None (extractive fallback) or a real response
        assert response is None or len(response) > 5


# ════════════════════════════════════════════════════════════════
# SECTION 11: Evaluation Data
# ════════════════════════════════════════════════════════════════

class TestEvaluation:
    def test_qa_pairs_exist(self):
        p = EVAL_DIR / "qa_pairs.json"
        assert p.exists()
        qa = json.loads(p.read_text())
        assert len(qa) >= 10

    def test_ablation_results_exist(self):
        p = EVAL_DIR / "ablation_results.json"
        assert p.exists()
        ab = json.loads(p.read_text())
        assert "fixed" in ab or "section_aware" in ab

    def test_ablation_metrics_range(self):
        p = EVAL_DIR / "ablation_results.json"
        ab = json.loads(p.read_text())
        for strategy, metrics in ab.items():
            for metric, val in metrics.items():
                if isinstance(val, float):
                    assert 0.0 <= val <= 1.0, f"{strategy}/{metric}={val} out of [0,1]"

    def test_ragas_results_exist(self):
        p = EVAL_DIR / "ragas_results.json"
        assert p.exists()
        rg = json.loads(p.read_text())
        assert "summary" in rg

    def test_ragas_metrics_range(self):
        p = EVAL_DIR / "ragas_results.json"
        rg = json.loads(p.read_text())
        s = rg["summary"]
        for metric in ["faithfulness", "answer_relevance", "context_precision"]:
            val = s.get(metric, -1)
            assert 0.0 <= val <= 1.0, f"RAGAS {metric}={val} out of [0,1]"


# ════════════════════════════════════════════════════════════════
# SECTION 12: Chunking Unit Tests
# ════════════════════════════════════════════════════════════════

class TestChunkingFunctions:
    def test_fixed_size_chunker(self):
        from src.chunk import fixed_size_chunker
        text = "This is a test sentence. " * 300
        meta = {"ticker": "TEST", "year": 2024, "section_type": "Risk Factors"}
        chunks = fixed_size_chunker(text, meta)
        assert len(chunks) > 0
        assert all(c["token_count"] <= 512 for c in chunks)

    def test_section_aware_chunker(self):
        from src.chunk import section_aware_chunker
        text = "Para one about risks.\n\nPara two about more risks.\n\nPara three."
        meta = {"ticker": "TEST", "year": 2024, "section_type": "Risk Factors"}
        chunks = section_aware_chunker(text, meta)
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)

    def test_fixed_respects_max_tokens(self):
        from src.chunk import fixed_size_chunker, count_tokens
        text = "risk " * 5000
        meta = {"ticker": "DEMO", "year": 2024, "section_type": "Risk Factors"}
        chunks = fixed_size_chunker(text, meta)
        for c in chunks:
            assert c["token_count"] <= 512

    def test_count_tokens(self):
        from src.chunk import count_tokens
        n = count_tokens("Hello, world!")
        assert isinstance(n, int)
        assert n > 0


# ════════════════════════════════════════════════════════════════
# SECTION 13: Z-Score Functions
# ════════════════════════════════════════════════════════════════

class TestZScoreFunctions:
    def test_get_val_empty_df(self):
        from src.zscore import get_val
        import pandas as pd
        result = get_val(pd.DataFrame(), ["Total Assets"])
        assert np.isnan(result)

    def test_fallback_row_always_returns(self):
        from src.zscore import _make_fallback_row
        row = _make_fallback_row("AAPL")
        assert row["ticker"] == "AAPL"
        assert row["z_score"] > 0
        assert row["zone"] in {"Safe", "Grey Zone", "Distress"}

    def test_zone_logic(self):
        from src.zscore import _make_fallback_row
        for ticker, (z, zone) in [("AAPL", (4.92, "Safe")),
                                   ("JPM", (1.95, "Grey Zone")),
                                   ("BA", (0.85, "Distress"))]:
            row = _make_fallback_row(ticker)
            assert row["zone"] == zone


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
