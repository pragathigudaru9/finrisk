"""
FinRisk — Full Pipeline Runner

Run all pipeline stages sequentially:
  Stage 3: Chunking
  Stage 4: Indexing (FAISS + BM25)
  Stage 6A: Sentiment (FinBERT)
  Stage 6B: Altman Z-Score
  Stage 7: Risk Intelligence (composite scores + YoY trends)
  Stage 8B: Evaluation (ablation + RAGAS)

Usage:
  cd finrisk/
  python run_pipeline.py [--stages 3 4 6a 6b 7 8]

Example (run all):
  python run_pipeline.py

Example (only stages 6–8):
  python run_pipeline.py --stages 6a 6b 7 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in path
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
        "--stages", nargs="+",
        default=["3", "4", "6a", "6b", "7", "8"],
        help="Stages to run (e.g. 3 4 6a 6b 7 8)",
    )
    args = parser.parse_args()
    stages = [s.lower() for s in args.stages]

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║            FinRisk — Full Pipeline Runner                  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Stages to run: {stages}")

    if "3" in stages:
        from src.chunk import build_all_chunks, run_acceptance_tests as chunk_tests
        run_stage("Stage 3: Chunking", build_all_chunks)
        chunk_tests()

    if "4" in stages:
        from src.index import build_all_indexes, test_retrieval, run_acceptance_tests as index_tests
        run_stage("Stage 4: Indexing", build_all_indexes)
        test_retrieval()
        index_tests()

    if "6a" in stages:
        from src.sentiment import build_sentiment_parquet
        run_stage("Stage 6A: Sentiment", build_sentiment_parquet)

    if "6b" in stages:
        from src.zscore import build_zscore_parquet
        run_stage("Stage 6B: Z-Score", build_zscore_parquet)

    if "7" in stages:
        from src.risk_score import build_risk_profiles, run_acceptance_tests as risk_tests
        scores_df, trends_df = run_stage("Stage 7: Risk Intelligence", build_risk_profiles)
        risk_tests(scores_df, trends_df)

    if "8" in stages:
        from src.evaluate import run_chunking_ablation, run_ragas_evaluation
        run_stage("Stage 8B-Ablation: Chunking Ablation", run_chunking_ablation)
        run_stage("Stage 8B-RAGAS: RAG Quality Eval", run_ragas_evaluation)

    logger.info("\n" + "="*70)
    logger.info("✅ ALL STAGES COMPLETE — Ready to run: streamlit run app.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()
