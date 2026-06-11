#!/usr/bin/env python3
"""
Quick verification script — checks every artifact is in place.
Run after pipeline completes.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, CHUNKS_DIR, MODELS_DIR, FINANCIALS_DIR, SENTIMENT_DIR, EVAL_DIR

OK = "✅"
FAIL = "❌"
WARN = "⚠️ "

def check(name, condition, detail=""):
    status = OK if condition else FAIL
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition

results = []

print("\n📦 DATA ARTIFACTS")
r = PROCESSED_DIR / "finrisk_sections.parquet"
results.append(check("finrisk_sections.parquet", r.exists(), f"{r.stat().st_size//1024//1024}MB" if r.exists() else "missing"))

r = PROCESSED_DIR / "risk_scores.parquet"
if r.exists():
    import pandas as pd
    df = pd.read_parquet(r)
    results.append(check("risk_scores.parquet", True, f"{len(df)} companies"))
else:
    results.append(check("risk_scores.parquet", False, "missing"))

r = PROCESSED_DIR / "yoy_trends.parquet"
if r.exists():
    import pandas as pd
    df = pd.read_parquet(r)
    results.append(check("yoy_trends.parquet", True, f"{len(df)} year-pairs"))
else:
    results.append(check("yoy_trends.parquet", False, "missing — run risk_score.py with BGE loaded"))

r = SENTIMENT_DIR / "sentiment_scores.parquet"
if r.exists():
    import pandas as pd
    df = pd.read_parquet(r)
    results.append(check("sentiment_scores.parquet", True, f"{len(df)} rows, {df['ticker'].nunique()} tickers"))
else:
    results.append(check("sentiment_scores.parquet", False, "missing"))

r = FINANCIALS_DIR / "zscore.parquet"
if r.exists():
    import pandas as pd
    df = pd.read_parquet(r)
    results.append(check("zscore.parquet", True, f"{len(df)} companies"))
else:
    results.append(check("zscore.parquet", False, "missing"))

print("\n🗂️  CHUNK FILES")
for strategy in ["fixed", "section_aware"]:
    p = CHUNKS_DIR / strategy / "chunks.jsonl"
    if p.exists():
        n = sum(1 for _ in open(p))
        results.append(check(f"chunks/{strategy}/chunks.jsonl", True, f"{n:,} chunks"))
    else:
        results.append(check(f"chunks/{strategy}/chunks.jsonl", False, "missing"))

print("\n🔍 INDEXES")
for strategy in ["fixed", "section_aware"]:
    faiss_p = MODELS_DIR / f"faiss_{strategy}" / "index.faiss"
    bm25_p = MODELS_DIR / f"bm25_{strategy}.pkl"
    if faiss_p.exists():
        import faiss
        idx = faiss.read_index(str(faiss_p))
        results.append(check(f"faiss_{strategy}", True, f"{idx.ntotal:,} vectors, {idx.d}d"))
    else:
        results.append(check(f"faiss_{strategy}/index.faiss", False, "missing"))
    results.append(check(f"bm25_{strategy}.pkl", bm25_p.exists()))

print("\n📊 EVAL FILES")
for fname in ["qa_pairs.json", "ablation_results.json", "ragas_results.json"]:
    p = EVAL_DIR / fname
    if p.exists():
        data = json.loads(p.read_text())
        results.append(check(fname, True, f"{len(data)} entries" if isinstance(data, list) else "loaded"))
    else:
        results.append(check(fname, False, "missing"))

passed = sum(results)
total = len(results)
print(f"\n{'='*50}")
print(f"TOTAL: {passed}/{total} checks passed")
if passed == total:
    print("🎉 ALL GOOD — Ready to run Streamlit!")
else:
    print(f"⚠️  {total - passed} artifacts missing — run missing pipeline stages")
print('='*50)
