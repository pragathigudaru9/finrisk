"""
FinRisk — Stage 2 (new): Zero-Shot Risk Category Extraction

Classifies filing chunks into 8 risk categories using facebook/bart-large-mnli.
No fine-tuning required — works out of the box via HuggingFace zero-shot pipeline.

Categories: Regulatory, Supply Chain, Demand, Liquidity,
            Cybersecurity, Legal, Credit, Geopolitical
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RISK_CATEGORIES, RISK_SPIKE_THRESHOLD, ZERO_SHOT_MODEL

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Classifier loader (singleton — load once, reuse)
# ──────────────────────────────────────────────────────────────

_classifier = None


def load_classifier():
    """Load zero-shot classification pipeline (lazy singleton)."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        logger.info(f"Loading zero-shot classifier: {ZERO_SHOT_MODEL}")
        _classifier = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL,
            device=-1,  # CPU; set to 0 for GPU
        )
        logger.info("Zero-shot classifier loaded.")
    return _classifier


# ──────────────────────────────────────────────────────────────
# Core classification
# ──────────────────────────────────────────────────────────────

def classify_chunks(chunks: list[dict], classifier=None) -> list[dict]:
    """
    Classify each chunk into one of 8 risk categories using zero-shot inference.

    Input:
        chunks: list of dicts with keys {chunk_id, text, ticker, period_label, section_type}
        classifier: optional pre-loaded pipeline (loaded lazily if None)

    Output:
        Same list, each dict augmented with:
          risk_category (str)   — top predicted label
          confidence    (float) — softmax score for top label, rounded to 4 dp
    """
    if classifier is None:
        classifier = load_classifier()

    results = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text.strip():
            results.append({**chunk, "risk_category": "Regulatory", "confidence": 0.0})
            continue

        # Truncate very long chunks to avoid slow inference
        truncated = text[:1024]
        try:
            out = classifier(truncated, RISK_CATEGORIES, multi_label=False)
            results.append({
                **chunk,
                "risk_category": out["labels"][0],
                "confidence": round(float(out["scores"][0]), 4),
            })
        except Exception as e:
            logger.warning(f"  Chunk {i} classification failed: {e}")
            results.append({**chunk, "risk_category": "Regulatory", "confidence": 0.0})

        if (i + 1) % 50 == 0:
            logger.info(f"  Classified {i + 1}/{len(chunks)} chunks")

    return results


# ──────────────────────────────────────────────────────────────
# Quarter-over-quarter counting
# ──────────────────────────────────────────────────────────────

def count_by_quarter(classified_chunks: list[dict]) -> dict:
    """
    Count risk category mentions per period and compute spike flags.

    Returns:
    {
      "Regulatory": {
        "current_quarter":  18,
        "previous_quarter": 5,
        "change_pct":       260.0,
        "flagged":          True    # True when change_pct >= RISK_SPIKE_THRESHOLD * 100
      },
      ...   (all 8 categories always present)
    }
    """
    current: dict[str, int] = {}
    previous: dict[str, int] = {}

    for chunk in classified_chunks:
        cat = chunk.get("risk_category", "")
        if not cat:
            continue
        period = chunk.get("period_label", "")
        if period == "current_quarter":
            current[cat] = current.get(cat, 0) + 1
        elif period == "previous_quarter":
            previous[cat] = previous.get(cat, 0) + 1

    result = {}
    for cat in RISK_CATEGORIES:
        curr = current.get(cat, 0)
        prev = previous.get(cat, 0)

        if prev > 0:
            pct = (curr - prev) / prev * 100
        elif curr > 0:
            pct = 100.0
        else:
            pct = 0.0

        result[cat] = {
            "current_quarter": curr,
            "previous_quarter": prev,
            "change_pct": round(pct, 1),
            "flagged": pct >= RISK_SPIKE_THRESHOLD * 100,
        }

    return result


# ──────────────────────────────────────────────────────────────
# Evidence extraction
# ──────────────────────────────────────────────────────────────

def extract_evidence(
    risk_category: str,
    classified_chunks: list[dict],
    period: str = "current_quarter",
    top_n: int = 3,
) -> list[str]:
    """
    Return top_n evidence sentences for a given risk category, sorted by
    classifier confidence (highest first).

    Args:
        risk_category: one of RISK_CATEGORIES
        classified_chunks: output of classify_chunks()
        period: "current_quarter" | "previous_quarter" | "annual"
        top_n: maximum results to return

    Returns:
        List of text strings (at most top_n items)
    """
    matching = [
        c for c in classified_chunks
        if c.get("risk_category") == risk_category
        and c.get("period_label") == period
    ]
    matching = sorted(matching, key=lambda x: x.get("confidence", 0.0), reverse=True)
    return [c["text"] for c in matching[:top_n]]
