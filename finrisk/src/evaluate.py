"""
FinRisk — Stage 8B: Evaluation Suite

Two evaluation tasks:
  1. Chunking ablation: Hit@5, Hit@10, MRR, NDCG for fixed vs section-aware
  2. RAG quality metrics: faithfulness, answer relevance, context precision
"""

import json
import sys
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_DIR, CHUNKS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Evaluation queries — matched to actual years in data (2023–2025)
# ──────────────────────────────────────────────────────────────
EVAL_QUERIES = [
    # Ticker-specific
    {"query": "What risks did Apple disclose in its Risk Factors?",
     "ticker_filter": "AAPL", "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "What did Microsoft say about cloud competition and AI?",
     "ticker_filter": "MSFT", "year_filter": None, "section_filter": "MD&A"},
    {"query": "How did Tesla describe manufacturing and supply chain risks?",
     "ticker_filter": "TSLA", "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "What credit risks did JPMorgan disclose?",
     "ticker_filter": "JPM", "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "What are NVIDIA's supply chain and semiconductor risks?",
     "ticker_filter": "NVDA", "year_filter": None, "section_filter": "Risk Factors"},

    # Comparative
    {"query": "How do Goldman Sachs and JPMorgan differ on credit risk?",
     "ticker_filter": ["GS", "JPM"], "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "Compare pharmaceutical risks between Pfizer and Eli Lilly",
     "ticker_filter": ["PFE", "LLY"], "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "How do ExxonMobil and Chevron discuss climate and regulatory risk?",
     "ticker_filter": ["XOM", "CVX"], "year_filter": None, "section_filter": "Risk Factors"},

    # Temporal
    {"query": "How did Tesla's risk disclosure evolve from 2023 to 2025?",
     "ticker_filter": "TSLA", "year_filter": [2023, 2024, 2025], "section_filter": "Risk Factors"},
    {"query": "How has Apple's risk profile changed over recent years?",
     "ticker_filter": "AAPL", "year_filter": [2023, 2024, 2025], "section_filter": "Risk Factors"},

    # Section-specific
    {"query": "What does Microsoft's MD&A say about AI investments?",
     "ticker_filter": "MSFT", "year_filter": None, "section_filter": "MD&A"},
    {"query": "How does Boeing discuss safety risks in its filings?",
     "ticker_filter": "BA", "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "What does Walmart's MD&A say about e-commerce strategy?",
     "ticker_filter": "WMT", "year_filter": None, "section_filter": "MD&A"},

    # Topic-specific (cross-company)
    {"query": "Which companies mention supply chain risk?",
     "ticker_filter": None, "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "Which companies discuss cybersecurity threats?",
     "ticker_filter": None, "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "What interest rate risks do banks disclose?",
     "ticker_filter": ["JPM", "BAC", "GS", "MS"], "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "How do energy companies discuss regulatory and climate risk?",
     "ticker_filter": ["XOM", "CVX"], "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "What does NVIDIA say about AI chip competition?",
     "ticker_filter": "NVDA", "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "What liquidity and leverage risks does Goldman Sachs disclose?",
     "ticker_filter": "GS", "year_filter": None, "section_filter": "Risk Factors"},
    {"query": "How does Johnson and Johnson discuss product liability risks?",
     "ticker_filter": "JNJ", "year_filter": None, "section_filter": "Risk Factors"},
]


def load_chunks(strategy: str) -> list[dict]:
    path = CHUNKS_DIR / strategy / "chunks.jsonl"
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def auto_annotate(queries: list[dict], chunks: list[dict]) -> list[dict]:
    """Annotate queries with relevant chunk IDs based on metadata filters."""
    annotated = []
    for q in queries:
        tf = q.get("ticker_filter")
        yf = q.get("year_filter")
        sf = q.get("section_filter")
        relevant = []
        for chunk in chunks:
            ok = True
            if tf:
                if isinstance(tf, list):
                    if chunk["ticker"] not in tf:
                        ok = False
                elif chunk["ticker"] != tf:
                    ok = False
            if yf and ok:
                if isinstance(yf, list):
                    if chunk["year"] not in yf:
                        ok = False
                elif chunk["year"] != yf:
                    ok = False
            if sf and ok:
                if chunk["section_type"] != sf:
                    ok = False
            if ok:
                relevant.append(chunk["chunk_id"])
        annotated.append({
            "query": q["query"],
            "relevant_ids": relevant[:20],
            "n_relevant": len(relevant),
        })
    return annotated


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set, k: int) -> float:
    """Compute NDCG@k."""
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def run_chunking_ablation() -> dict:
    """Chunking ablation: compare fixed vs section-aware retrieval."""
    logger.info("=" * 60)
    logger.info("CHUNKING ABLATION STUDY")
    logger.info("=" * 60)

    from src.retrieve import HybridRetriever

    results = {}

    for strategy in ["fixed", "section_aware"]:
        logger.info(f"\nEvaluating strategy: {strategy}")
        chunks = load_chunks(strategy)
        annotated = auto_annotate(EVAL_QUERIES, chunks)
        retriever = HybridRetriever(strategy)

        hit5, hit10, mrr, ndcg5, ndcg10 = [], [], [], [], []

        for q, gt in zip(EVAL_QUERIES, annotated):
            if not gt["relevant_ids"]:
                continue

            gt_set = set(gt["relevant_ids"])
            retrieved = retriever.retrieve(q["query"], top_k=10)
            ids = [r["chunk_id"] for r in retrieved]

            h5 = 1.0 if set(ids[:5]) & gt_set else 0.0
            h10 = 1.0 if set(ids[:10]) & gt_set else 0.0

            rr = 0.0
            for rank, rid in enumerate(ids, 1):
                if rid in gt_set:
                    rr = 1.0 / rank
                    break

            hit5.append(h5)
            hit10.append(h10)
            mrr.append(rr)
            ndcg5.append(ndcg_at_k(ids, gt_set, 5))
            ndcg10.append(ndcg_at_k(ids, gt_set, 10))

            logger.info(f"  Hit@5={h5:.0f} Hit@10={h10:.0f} MRR={rr:.3f} — {q['query'][:50]}")

        results[strategy] = {
            "Hit@5": round(float(np.mean(hit5)) if hit5 else 0.0, 3),
            "Hit@10": round(float(np.mean(hit10)) if hit10 else 0.0, 3),
            "MRR": round(float(np.mean(mrr)) if mrr else 0.0, 3),
            "NDCG@5": round(float(np.mean(ndcg5)) if ndcg5 else 0.0, 3),
            "NDCG@10": round(float(np.mean(ndcg10)) if ndcg10 else 0.0, 3),
            "n_queries": len(hit5),
        }

    # Also run BM25-only and FAISS-only for Table 1 in the paper
    logger.info("\nEvaluating BM25-only and FAISS-only for ablation...")
    try:
        import faiss, pickle
        from sentence_transformers import SentenceTransformer
        from config import MODELS_DIR, BGE_EMBEDDING_MODEL, TOP_K_RETRIEVAL

        strategy = "section_aware"
        chunks = load_chunks(strategy)
        annotated = auto_annotate(EVAL_QUERIES, chunks)

        # Load FAISS
        faiss_dir = MODELS_DIR / f"faiss_{strategy}"
        faiss_index = faiss.read_index(str(faiss_dir / "index.faiss"))
        with open(faiss_dir / "chunks_meta.json") as f:
            faiss_meta = json.load(f)
        model = SentenceTransformer(BGE_EMBEDDING_MODEL)

        # Load BM25
        with open(MODELS_DIR / f"bm25_{strategy}.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        bm25_map = bm25_data["mapping"]

        for method in ["bm25_only", "faiss_only"]:
            hit5, mrr = [], []
            for q, gt in zip(EVAL_QUERIES, annotated):
                if not gt["relevant_ids"]:
                    continue
                gt_set = set(gt["relevant_ids"])

                if method == "faiss_only":
                    q_emb = np.array(model.encode([q["query"]], normalize_embeddings=True), dtype=np.float32)
                    _, idxs = faiss_index.search(q_emb, 5)
                    ids = [faiss_meta[i]["chunk_id"] for i in idxs[0] if i >= 0]
                else:
                    tokens = q["query"].lower().split()
                    scores = bm25.get_scores(tokens)
                    top_idxs = np.argsort(scores)[::-1][:5]
                    ids = [bm25_map[i]["chunk_id"] for i in top_idxs]

                hit5.append(1.0 if set(ids[:5]) & gt_set else 0.0)
                rr = 0.0
                for rank, rid in enumerate(ids, 1):
                    if rid in gt_set:
                        rr = 1.0 / rank
                        break
                mrr.append(rr)

            results[method] = {
                "Hit@5": round(float(np.mean(hit5)) if hit5 else 0.0, 3),
                "MRR": round(float(np.mean(mrr)) if mrr else 0.0, 3),
                "n_queries": len(hit5),
            }
    except Exception as e:
        logger.warning(f"BM25/FAISS-only evaluation skipped: {e}")

    # Print ablation table
    print("\n" + "=" * 70)
    print("CHUNKING + RETRIEVAL ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8} {'NDCG@5':>8} {'NDCG@10':>8}")
    print("-" * 65)
    for method, m in results.items():
        print(
            f"{method:<20} {m.get('Hit@5', '-'):>8} {m.get('Hit@10', '-'):>8} "
            f"{m.get('MRR', '-'):>8} {m.get('NDCG@5', '-'):>8} {m.get('NDCG@10', '-'):>8}"
        )

    # Save
    out = EVAL_DIR / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved ablation results to {out}")
    return results


QA_PAIRS = [
    {"question": "What liquidity risks did Goldman Sachs disclose?",
     "ground_truth": "Goldman Sachs faces risks from market funding conditions, counterparty credit, and potential cash flow disruptions."},
    {"question": "What cybersecurity risks does Microsoft face?",
     "ground_truth": "Microsoft faces risks from cyberattacks, data breaches, ransomware, and cloud infrastructure vulnerabilities."},
    {"question": "How does Apple describe supply chain concentration risk?",
     "ground_truth": "Apple depends on single or limited suppliers for certain components, making it vulnerable to supply disruptions."},
    {"question": "What regulatory risks does JPMorgan Chase disclose?",
     "ground_truth": "JPMorgan faces evolving financial regulations, capital requirements, and compliance obligations across jurisdictions."},
    {"question": "How does Tesla describe competition in the EV market?",
     "ground_truth": "Tesla faces increasing competition from traditional automakers and new EV companies across all vehicle segments."},
    {"question": "What climate-related risks does ExxonMobil discuss?",
     "ground_truth": "ExxonMobil discusses transition risks, physical climate risks, and market shifts toward lower-carbon energy."},
    {"question": "What are NVIDIA's key semiconductor supply risks?",
     "ground_truth": "NVIDIA faces risks from semiconductor supply shortages, manufacturing dependencies, and competitive pressure."},
    {"question": "How does Bank of America discuss interest rate risk?",
     "ground_truth": "Bank of America discloses sensitivity to rate changes affecting net interest income and investment portfolios."},
    {"question": "What product liability risks does Johnson Johnson face?",
     "ground_truth": "J&J faces product liability risks from ongoing litigation related to pharmaceutical and consumer health products."},
    {"question": "How does Boeing discuss safety and certification risks?",
     "ground_truth": "Boeing discloses risks from aircraft safety investigations, FAA certification, and production quality oversight."},
    {"question": "What geopolitical risks does Google Alphabet disclose?",
     "ground_truth": "Alphabet discusses international regulations, data localization requirements, and geopolitical tensions."},
    {"question": "How does Pfizer discuss drug pipeline and patent risks?",
     "ground_truth": "Pfizer discloses risks from clinical trial outcomes, approval timelines, and patent expirations."},
    {"question": "What does Eli Lilly say about drug pricing risks?",
     "ground_truth": "Eli Lilly discusses government pricing negotiations, insurance coverage changes, and pharma pricing pressure."},
    {"question": "How does Chevron discuss environmental compliance risks?",
     "ground_truth": "Chevron discloses risks from environmental regulations, remediation obligations, and operational liability."},
    {"question": "What acquisition risks does Microsoft disclose?",
     "ground_truth": "Microsoft discusses integration challenges, regulatory approval uncertainty, and business impact from acquisitions."},
]


def run_ragas_evaluation() -> dict:
    """RAG quality evaluation on QA pairs."""
    logger.info("=" * 60)
    logger.info("RAG QUALITY EVALUATION (RAGAS-style)")
    logger.info("=" * 60)

    from src.rag import FinRiskRAG
    rag = FinRiskRAG()

    qa_path = EVAL_DIR / "qa_pairs.json"
    with open(qa_path, "w") as f:
        json.dump(QA_PAIRS, f, indent=2)

    results = []
    stop = {"what", "how", "did", "does", "the", "a", "an", "in", "of", "to",
            "and", "or", "is", "are", "was", "were", "for", "with", "that", "this"}

    for i, qa in enumerate(QA_PAIRS, 1):
        logger.info(f"[{i}/{len(QA_PAIRS)}] {qa['question'][:55]}...")
        try:
            resp = rag.query(qa["question"])
            answer = resp["answer"]
            chunks = resp["retrieved_chunks"]

            answer_text = answer.get("summary", "")
            chunk_text = " ".join(c["text"] for c in chunks)

            aw = set(answer_text.lower().split()) - stop
            cw = set(chunk_text.lower().split()) - stop
            qw = set(qa["question"].lower().split()) - stop
            gw = set(qa["ground_truth"].lower().split()) - stop

            faithfulness = min(len(aw & cw) / len(aw), 1.0) if aw else 0.0
            relevance = min(len(aw & qw) / len(qw), 1.0) if qw else 0.0
            precision = min(len(cw & gw) / len(gw), 1.0) if gw else 0.0

            results.append({
                "question": qa["question"],
                "faithfulness": round(faithfulness, 4),
                "answer_relevance": round(relevance, 4),
                "context_precision": round(precision, 4),
            })
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                "question": qa["question"],
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
            })

    avg_f = float(np.mean([r["faithfulness"] for r in results]))
    avg_r = float(np.mean([r["answer_relevance"] for r in results]))
    avg_p = float(np.mean([r["context_precision"] for r in results]))

    print("\n" + "=" * 60)
    print("RAG QUALITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Score':>8} {'Target':>8}")
    print("-" * 43)
    print(f"{'Faithfulness':<25} {avg_f:>8.3f} {'≥ 0.85':>8}")
    print(f"{'Answer Relevance':<25} {avg_r:>8.3f} {'≥ 0.80':>8}")
    print(f"{'Context Precision':<25} {avg_p:>8.3f} {'≥ 0.80':>8}")

    eval_out = {
        "summary": {
            "faithfulness": avg_f,
            "answer_relevance": avg_r,
            "context_precision": avg_p,
            "num_queries": len(results),
        },
        "per_query": results,
    }

    out = EVAL_DIR / "ragas_results.json"
    with open(out, "w") as f:
        json.dump(eval_out, f, indent=2)
    logger.info(f"Saved RAG evaluation results to {out}")
    return eval_out


if __name__ == "__main__":
    ablation = run_chunking_ablation()
    ragas = run_ragas_evaluation()


# ──────────────────────────────────────────────────────────────
# Risk extraction accuracy evaluation (new)
# ──────────────────────────────────────────────────────────────

def evaluate_risk_extraction(
    labeled_path: str | None = None,
) -> dict:
    """
    Measure zero-shot classifier accuracy against hand-labeled filing chunks.

    labeled_path: path to JSON file with list of
        {"text": "...", "true_label": "Regulatory"}
        If None, uses data/eval/labeled_risk_chunks.json by default.

    Returns: sklearn classification_report as a dict.
    Prints the full text report.
    Requires >= 75% accuracy to pass the acceptance test.
    """
    import json
    from sklearn.metrics import classification_report, accuracy_score

    if labeled_path is None:
        labeled_path = str(EVAL_DIR / "labeled_risk_chunks.json")

    p = Path(labeled_path)
    if not p.exists():
        raise FileNotFoundError(f"Labeled data not found: {p}")

    with open(p) as f:
        labeled = json.load(f)

    logger.info(f"Evaluating zero-shot classifier on {len(labeled)} labeled chunks...")

    texts = [item["text"] for item in labeled]
    true_labels = [item["true_label"] for item in labeled]

    from src.extract_risks import load_classifier, RISK_CATEGORIES
    clf = load_classifier()

    pred_labels = []
    for i, text in enumerate(texts):
        result = clf(text[:1024], RISK_CATEGORIES, multi_label=False)
        pred_labels.append(result["labels"][0])
        if (i + 1) % 10 == 0:
            logger.info(f"  Evaluated {i + 1}/{len(texts)}")

    acc = accuracy_score(true_labels, pred_labels)
    report_str = classification_report(
        true_labels, pred_labels, zero_division=0
    )
    report_dict = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )

    print("\n" + "=" * 60)
    print("RISK EXTRACTION ACCURACY EVALUATION")
    print("=" * 60)
    print(report_str)
    print(f"Overall Accuracy: {acc:.1%}")
    if acc >= 0.75:
        print("✓ PASS: Accuracy >= 75%")
    else:
        print(f"⚠ WARN: Accuracy {acc:.1%} below 75% target")
    print("=" * 60)

    # Save results alongside other eval files
    out = EVAL_DIR / "risk_extraction_results.json"
    with open(out, "w") as f:
        json.dump({"accuracy": acc, "report": report_dict}, f, indent=2)
    logger.info(f"Saved risk extraction evaluation to {out}")

    return report_dict

