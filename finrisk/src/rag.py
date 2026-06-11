"""
FinRisk AI — Stage 8A: Conversational RAG Pipeline

Pipeline:
  User query → Hybrid retrieval (top 5 chunks) → Groq LLM → Structured JSON answer

Supports:
  - Groq (llama-3.3-70b-versatile) via API
  - Extractive fallback (no LLM needed)
"""

import json
import re
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


RAG_SYSTEM_PROMPT = """You are a financial risk analyst specializing in SEC 10-K filing analysis.
Answer the user's question using ONLY the provided filing excerpts.
For every claim, cite the exact source: [TICKER, YEAR, SECTION].
If the excerpts do not contain enough information, say so explicitly.
Do NOT speculate or add information not present in the excerpts.

Respond in this exact JSON format (no markdown, no extra text):
{
  "summary": "2-3 sentence risk summary grounded in the excerpts",
  "key_risks": ["risk 1 with citation [TICKER, YEAR]", "risk 2 with citation", ...],
  "citations": [{"claim": "exact quote or paraphrase", "source": "TICKER, YEAR, SECTION"}],
  "confidence": "high/medium/low"
}"""


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks into the LLM prompt context."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"---\n[Excerpt {i}]\n"
            f"Source: {chunk['ticker']}, {chunk['year']}, {chunk['section_type']}\n"
            f"Text: {chunk['text'][:2000]}\n"
            f"---"
        )
    return "\n".join(parts)


def call_groq(system_prompt: str, user_prompt: str) -> str | None:
    """Call Groq LLM API."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return None


def extractive_fallback(query: str, chunks: list[dict]) -> dict:
    """
    Fallback when no LLM is available: extract key sentences from chunks
    and construct a structured response.
    """
    stop_words = {
        "what", "how", "did", "does", "the", "a", "an", "in", "of", "to",
        "and", "or", "is", "are", "was", "were", "for", "with", "that", "this",
        "do", "has", "have", "had", "which", "their", "by", "from", "at",
    }
    query_words = set(w.lower() for w in re.findall(r'\w+', query)) - stop_words

    all_sentences = []
    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
        for sent in sentences:
            if len(sent.split()) < 6:
                continue
            sent_words = set(w.lower() for w in re.findall(r'\w+', sent))
            overlap = len(query_words & sent_words)
            all_sentences.append({
                "text": sent.strip(),
                "score": overlap,
                "ticker": chunk["ticker"],
                "year": chunk["year"],
                "section_type": chunk["section_type"],
            })

    all_sentences.sort(key=lambda x: x["score"], reverse=True)
    top = all_sentences[:5]

    summary_parts = [s["text"][:200] for s in top[:2]]
    summary = " ".join(summary_parts) if summary_parts else "Insufficient information in retrieved excerpts."

    return {
        "summary": summary,
        "key_risks": [s["text"][:150] for s in top[:4]],
        "citations": [
            {
                "claim": s["text"][:120],
                "source": f"{s['ticker']}, {s['year']}, {s['section_type']}"
            }
            for s in top[:4]
        ],
        "confidence": "medium" if top else "low",
    }


def parse_llm_json(raw: str) -> dict | None:
    """Robustly parse JSON from LLM output."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return None


class FinRiskRAG:
    """
    Conversational RAG pipeline for financial risk queries.
    Uses Groq Llama-3.3-70b with extractive fallback.
    """

    def __init__(self, strategy: str = "section_aware"):
        """Load retriever and LLM client."""
        from src.retrieve import HybridRetriever

        self.retriever = HybridRetriever(strategy)
        self.strategy = strategy

        self.use_llm = bool(GROQ_API_KEY)
        if self.use_llm:
            logger.info(f"RAG initialized with Groq ({GROQ_MODEL})")
        else:
            logger.info("RAG initialized with extractive fallback (no GROQ_API_KEY)")

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks
        2. Generate answer (Groq LLM or extractive)
        3. Return structured response
        """
        # Step 1: Retrieve
        chunks = self.retriever.retrieve(question, top_k=top_k)

        # Step 2: Generate answer
        if self.use_llm and chunks:
            context = format_chunks_for_prompt(chunks)
            user_prompt = f"FILING EXCERPTS:\n{context}\n\nQUESTION: {question}"

            raw_response = call_groq(RAG_SYSTEM_PROMPT, user_prompt)

            if raw_response:
                parsed = parse_llm_json(raw_response)
                if parsed:
                    answer = parsed
                else:
                    # JSON parse failed — wrap raw
                    answer = {
                        "summary": raw_response[:600],
                        "key_risks": [],
                        "citations": [],
                        "confidence": "medium",
                        "_warning": "JSON parsing failed, raw response returned",
                    }
            else:
                answer = extractive_fallback(question, chunks)
        else:
            answer = extractive_fallback(question, chunks)

        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "retrieval_strategy": self.strategy,
            "num_chunks": len(chunks),
        }


if __name__ == "__main__":
    rag = FinRiskRAG()

    test_queries = [
        "What are Apple's biggest supply chain risks?",
        "How does Tesla describe competition in the EV market?",
        "What credit risks did JPMorgan disclose in 2024?",
        "What cybersecurity risks does Microsoft mention?",
        "How do energy companies discuss climate change risk?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print(f"{'='*70}")

        result = rag.query(q)
        answer = result["answer"]

        print(f"\nSummary: {answer.get('summary', 'N/A')}")
        print(f"\nKey Risks:")
        for r in answer.get("key_risks", []):
            print(f"  • {r[:150]}")
        print(f"\nCitations:")
        for c in answer.get("citations", []):
            print(f"  [{c.get('source', '?')}] {c.get('claim', '')[:100]}")
        print(f"\nConfidence: {answer.get('confidence', 'N/A')}")
        print(f"Retrieved: {result['num_chunks']} chunks")
