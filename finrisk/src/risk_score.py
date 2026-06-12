"""
FinRisk — Risk Summary Layer

Single-company risk summary built from zero-shot classified filing chunks.
Replaces the old composite weighted scoring model.

Exports:
  get_risk_summary(ticker, classified_chunks) -> dict
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RISK_CATEGORIES

logger = logging.getLogger(__name__)


def get_risk_summary(ticker: str, classified_chunks: list[dict]) -> dict:
    """
    Build a complete single-company risk summary from classified filing chunks.

    Args:
        ticker: stock ticker (e.g. "AAPL")
        classified_chunks: output of extract_risks.classify_chunks()

    Returns:
    {
        "ticker": "AAPL",
        "risk_counts": {
            "Regulatory": {
                "current_quarter": 18,
                "previous_quarter": 5,
                "change_pct": 260.0,
                "flagged": True
            },
            ...  (all 8 categories always present)
        },
        "flagged_categories": ["Regulatory"],
        "evidence": {
            "Regulatory": ["sentence 1", "sentence 2", "sentence 3"]
        }
    }
    """
    from src.extract_risks import count_by_quarter, extract_evidence

    risk_counts = count_by_quarter(classified_chunks)
    flagged = [cat for cat, d in risk_counts.items() if d["flagged"]]
    evidence = {cat: extract_evidence(cat, classified_chunks) for cat in flagged}

    return {
        "ticker": ticker,
        "risk_counts": risk_counts,
        "flagged_categories": flagged,
        "evidence": evidence,
    }
