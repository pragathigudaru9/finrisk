"""
FinRisk — Pipeline Runner

Run pipeline stages for a single company or for evaluation.

Usage:
  cd finrisk/
  python run_pipeline.py --ticker AAPL
  python run_pipeline.py --ticker NVDA --stages 3 4
  python run_pipeline.py --stages 8        # evaluation only (no ticker needed)

Stages:
  3    Chunking (fixed + section-aware)
  4    Indexing (FAISS + BM25)
  6a   Sentiment (FinBERT)
  6b   Altman Z-Score (yfinance)
  8    Evaluation (chunking ablation + RAGAS + risk extraction accuracy)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_stage(name: str, fn, *args, **kwargs):
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {name}")
    logger.info(f"{'='*70}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"✓ {name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        logger.error(f"✗ {name} FAILED: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="FinRisk Pipeline Runner")
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Company ticker to analyze (e.g. AAPL, NVDA). Required for stages 1, 6a, 6b.",
    )
    parser.add_argument(
        "--stages", nargs="+",
        default=["3", "4", "6a", "6b", "8"],
        help="Stages to run (e.g. 3 4 6a 6b 8). Stage 1 requires --ticker.",
    )
    args = parser.parse_args()
    stages = [s.lower() for s in args.stages]
    ticker = args.ticker.upper() if args.ticker else None

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║          FinRisk — Single-Company Pipeline Runner         ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    if ticker:
        logger.info(f"Company: {ticker}")
    logger.info(f"Stages:  {stages}")

    # Stage 1: Single-company ingestion
    if "1" in stages:
        if not ticker:
            logger.error("Stage 1 requires --ticker (e.g. --ticker AAPL)")
            sys.exit(1)
        from src.ingest import download_company_latest
        sections_df = run_stage(
            f"Stage 1: Ingest {ticker}", download_company_latest, ticker
        )
        logger.info(f"  Got {len(sections_df)} sections for {ticker}")

    # Stage 3: Chunking (builds from existing parquet if available)
    if "3" in stages:
        from src.chunk import build_all_chunks, run_acceptance_tests as chunk_tests
        run_stage("Stage 3: Chunking", build_all_chunks)
        chunk_tests()

    # Stage 4: Indexing
    if "4" in stages:
        from src.index import build_all_indexes, test_retrieval, run_acceptance_tests as index_tests
        run_stage("Stage 4: Indexing", build_all_indexes)
        test_retrieval()
        index_tests()

    # Stage 6A: FinBERT Sentiment
    if "6a" in stages:
        from src.sentiment import build_sentiment_parquet
        run_stage("Stage 6A: Sentiment", build_sentiment_parquet)

    # Stage 6B: Altman Z-Score
    if "6b" in stages:
        from src.zscore import build_zscore_parquet
        run_stage("Stage 6B: Z-Score", build_zscore_parquet)

    # Stage 8: Evaluation
    if "8" in stages:
        from src.evaluate import run_chunking_ablation, run_ragas_evaluation, evaluate_risk_extraction
        run_stage("Stage 8A: Chunking Ablation", run_chunking_ablation)
        run_stage("Stage 8B: RAGAS Evaluation", run_ragas_evaluation)
        labeled_path = Path(__file__).parent / "data" / "eval" / "labeled_risk_chunks.json"
        if labeled_path.exists():
            run_stage("Stage 8C: Risk Extraction Accuracy", evaluate_risk_extraction)
        else:
            logger.warning("Skipping Stage 8C: labeled_risk_chunks.json not found")

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL STAGES COMPLETE — Run: streamlit run app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
