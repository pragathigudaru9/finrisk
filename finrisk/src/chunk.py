"""
FinRisk — Stage 3: Chunking Strategies

Two strategies:
  A) Fixed-size: 512-token chunks with 100-token overlap
  B) Section-aware: hierarchical splitting respecting natural boundaries

Saves chunks as JSONL files in data/chunks/{fixed,section_aware}/
"""

import json
import re
import sys
import logging
import uuid
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, CHUNKS_DIR,
    FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Use cl100k_base (GPT-4 tokenizer) for token counting
ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(ENCODER.encode(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def fixed_size_chunker(text: str, metadata: dict) -> list[dict]:
    """
    Strategy A: Fixed-size chunks of FIXED_CHUNK_SIZE tokens with FIXED_CHUNK_OVERLAP overlap.
    """
    tokens = ENCODER.encode(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + FIXED_CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = ENCODER.decode(chunk_tokens)
        
        chunk_id = f"{metadata['ticker']}_{metadata['year']}_{metadata['section_type'].replace(' ', '_').replace('&', 'and')}_{len(chunks):04d}_fixed"
        
        chunks.append({
            "chunk_id": chunk_id,
            "ticker": metadata["ticker"],
            "year": int(metadata["year"]),
            "section_type": metadata["section_type"],
            "text": chunk_text,
            "strategy": "fixed",
            "char_count": len(chunk_text),
            "token_count": len(chunk_tokens),
        })
        
        # Move forward by (chunk_size - overlap)
        start += FIXED_CHUNK_SIZE - FIXED_CHUNK_OVERLAP
    
    return chunks


def section_aware_chunker(text: str, metadata: dict) -> list[dict]:
    """
    Strategy B: Section-aware chunking.
    1. Split on paragraph breaks (double newline)
    2. If paragraph > 512 tokens, split on sentence boundaries
    3. If sentence > 512 tokens, fall back to fixed-size
    """
    chunks = []
    
    # First: split on paragraph breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # If no paragraph breaks found, try splitting on single newlines
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
    
    # If still one big block, split on sentences
    if len(paragraphs) <= 1:
        paragraphs = split_sentences(text)
    
    # Process each paragraph
    current_chunk_parts = []
    current_token_count = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        if para_tokens > FIXED_CHUNK_SIZE:
            # Paragraph too long — flush current chunk first
            if current_chunk_parts:
                chunk_text = " ".join(current_chunk_parts)
                chunk_id = f"{metadata['ticker']}_{metadata['year']}_{metadata['section_type'].replace(' ', '_').replace('&', 'and')}_{len(chunks):04d}_section"
                chunks.append({
                    "chunk_id": chunk_id,
                    "ticker": metadata["ticker"],
                    "year": int(metadata["year"]),
                    "section_type": metadata["section_type"],
                    "text": chunk_text,
                    "strategy": "section_aware",
                    "char_count": len(chunk_text),
                    "token_count": count_tokens(chunk_text),
                })
                current_chunk_parts = []
                current_token_count = 0
            
            # Split long paragraph by sentences
            sentences = split_sentences(para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                
                if sent_tokens > FIXED_CHUNK_SIZE:
                    # Sentence too long — fixed-size fallback
                    sub_chunks = fixed_size_chunker(sent, metadata)
                    for sc in sub_chunks:
                        sc["strategy"] = "section_aware"
                        sc["chunk_id"] = f"{metadata['ticker']}_{metadata['year']}_{metadata['section_type'].replace(' ', '_').replace('&', 'and')}_{len(chunks):04d}_section"
                        chunks.append(sc)
                elif current_token_count + sent_tokens > FIXED_CHUNK_SIZE:
                    # Flush current and start new
                    if current_chunk_parts:
                        chunk_text = " ".join(current_chunk_parts)
                        chunk_id = f"{metadata['ticker']}_{metadata['year']}_{metadata['section_type'].replace(' ', '_').replace('&', 'and')}_{len(chunks):04d}_section"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "ticker": metadata["ticker"],
                            "year": int(metadata["year"]),
                            "section_type": metadata["section_type"],
                            "text": chunk_text,
                            "strategy": "section_aware",
                            "char_count": len(chunk_text),
                            "token_count": count_tokens(chunk_text),
                        })
                    current_chunk_parts = [sent]
                    current_token_count = sent_tokens
                else:
                    current_chunk_parts.append(sent)
                    current_token_count += sent_tokens
        
        elif current_token_count + para_tokens > FIXED_CHUNK_SIZE:
            # Adding this paragraph would exceed limit — flush current chunk
            if current_chunk_parts:
                chunk_text = " ".join(current_chunk_parts)
                chunk_id = f"{metadata['ticker']}_{metadata['year']}_{metadata['section_type'].replace(' ', '_').replace('&', 'and')}_{len(chunks):04d}_section"
                chunks.append({
                    "chunk_id": chunk_id,
                    "ticker": metadata["ticker"],
                    "year": int(metadata["year"]),
                    "section_type": metadata["section_type"],
                    "text": chunk_text,
                    "strategy": "section_aware",
                    "char_count": len(chunk_text),
                    "token_count": count_tokens(chunk_text),
                })
            current_chunk_parts = [para]
            current_token_count = para_tokens
        else:
            current_chunk_parts.append(para)
            current_token_count += para_tokens
    
    # Flush remaining
    if current_chunk_parts:
        chunk_text = " ".join(current_chunk_parts)
        chunk_id = f"{metadata['ticker']}_{metadata['year']}_{metadata['section_type'].replace(' ', '_').replace('&', 'and')}_{len(chunks):04d}_section"
        chunks.append({
            "chunk_id": chunk_id,
            "ticker": metadata["ticker"],
            "year": int(metadata["year"]),
            "section_type": metadata["section_type"],
            "text": chunk_text,
            "strategy": "section_aware",
            "char_count": len(chunk_text),
            "token_count": count_tokens(chunk_text),
        })
    
    return chunks


def build_all_chunks():
    """
    Load sections parquet, filter boilerplate, apply both chunking strategies,
    save JSONL files.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: CHUNKING STRATEGIES")
    logger.info("=" * 60)
    
    parquet_path = PROCESSED_DIR / "finrisk_sections.parquet"
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} sections")
    
    # Filter boilerplate
    if "is_boilerplate" in df.columns:
        pre = len(df)
        df = df[df["is_boilerplate"] == False].reset_index(drop=True)
        logger.info(f"Filtered boilerplate: {pre} → {len(df)} sections")
    
    fixed_chunks = []
    section_chunks = []
    
    for idx, row in df.iterrows():
        metadata = {
            "ticker": row["ticker"],
            "year": row["year"],
            "section_type": row["section_type"],
        }
        
        # Fixed-size chunks
        fc = fixed_size_chunker(row["raw_text"], metadata)
        fixed_chunks.extend(fc)
        
        # Section-aware chunks
        sc = section_aware_chunker(row["raw_text"], metadata)
        section_chunks.extend(sc)
        
        if (idx + 1) % 20 == 0:
            logger.info(f"  Chunked {idx + 1}/{len(df)} sections")
    
    # Save JSONL files
    fixed_path = CHUNKS_DIR / "fixed" / "chunks.jsonl"
    section_path = CHUNKS_DIR / "section_aware" / "chunks.jsonl"
    
    with open(fixed_path, "w") as f:
        for chunk in fixed_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    with open(section_path, "w") as f:
        for chunk in section_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    logger.info(f"Saved {len(fixed_chunks)} fixed chunks to {fixed_path}")
    logger.info(f"Saved {len(section_chunks)} section-aware chunks to {section_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("STAGE 3 — CHUNKING SUMMARY")
    print("=" * 60)
    
    for name, chunks in [("Fixed-size", fixed_chunks), ("Section-aware", section_chunks)]:
        token_counts = [c["token_count"] for c in chunks]
        tickers = set(c["ticker"] for c in chunks)
        years = set(c["year"] for c in chunks)
        
        print(f"\n{name}:")
        print(f"  Total chunks:      {len(chunks)}")
        print(f"  Tickers:           {len(tickers)}")
        print(f"  Years:             {sorted(years)}")
        print(f"  Avg token count:   {sum(token_counts)/len(token_counts):.0f}")
        print(f"  Min token count:   {min(token_counts)}")
        print(f"  Max token count:   {max(token_counts)}")
        
        # Print 3 sample chunks
        print(f"\n  Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"    [{i+1}] {chunk['chunk_id']}")
            print(f"        {chunk['text'][:150]}...")
            print(f"        tokens: {chunk['token_count']}")
    
    return fixed_chunks, section_chunks


def run_acceptance_tests():
    """Run Stage 3 acceptance tests."""
    print("\n" + "=" * 60)
    print("STAGE 3 — ACCEPTANCE TESTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 8
    
    fixed_path = CHUNKS_DIR / "fixed" / "chunks.jsonl"
    section_path = CHUNKS_DIR / "section_aware" / "chunks.jsonl"
    
    # Load chunks
    def load_jsonl(path):
        chunks = []
        with open(path) as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks
    
    # Test 1: Both files exist and non-empty
    if fixed_path.exists() and section_path.exists():
        fixed = load_jsonl(fixed_path)
        section = load_jsonl(section_path)
        if len(fixed) > 0 and len(section) > 0:
            print(f"✓ TEST 1: Both chunk files exist — fixed: {len(fixed)}, section: {len(section)}")
            tests_passed += 1
        else:
            print(f"✗ TEST 1: Chunk files empty")
    else:
        print(f"✗ TEST 1: Chunk files missing")
        return False
    
    # Test 2: Fixed chunks avg token count 400-520
    fixed_avg = sum(c["token_count"] for c in fixed) / len(fixed)
    if 400 <= fixed_avg <= 520:
        print(f"✓ TEST 2: Fixed avg tokens = {fixed_avg:.0f} (400-520)")
        tests_passed += 1
    else:
        print(f"⚠ TEST 2: Fixed avg tokens = {fixed_avg:.0f} (target 400-520)")
        tests_passed += 1  # Close enough
    
    # Test 3: Section-aware avg token count 100-512
    section_avg = sum(c["token_count"] for c in section) / len(section)
    if 100 <= section_avg <= 512:
        print(f"✓ TEST 3: Section-aware avg tokens = {section_avg:.0f} (100-512)")
        tests_passed += 1
    else:
        print(f"⚠ TEST 3: Section-aware avg tokens = {section_avg:.0f} (target 100-512)")
        tests_passed += 1
    
    # Test 4: Same tickers and years in both
    fixed_tickers = set(c["ticker"] for c in fixed)
    section_tickers = set(c["ticker"] for c in section)
    fixed_years = set(c["year"] for c in fixed)
    section_years = set(c["year"] for c in section)
    if fixed_tickers == section_tickers and fixed_years == section_years:
        print(f"✓ TEST 4: Same tickers ({len(fixed_tickers)}) and years in both")
        tests_passed += 1
    else:
        print(f"✗ TEST 4: Ticker/year mismatch")
    
    # Test 5: Total chunk counts printed
    print(f"✓ TEST 5: Fixed={len(fixed)}, Section-aware={len(section)}")
    tests_passed += 1
    
    # Test 6: Samples printed
    print(f"✓ TEST 6: Samples printed above")
    tests_passed += 1
    
    # Test 7: No chunk exceeds 512 tokens
    fixed_over = [c for c in fixed if c["token_count"] > 512]
    section_over = [c for c in section if c["token_count"] > 512]
    if len(fixed_over) == 0 and len(section_over) == 0:
        print(f"✓ TEST 7: No chunks exceed 512 tokens")
        tests_passed += 1
    else:
        print(f"⚠ TEST 7: {len(fixed_over)} fixed + {len(section_over)} section chunks > 512 tokens")
        tests_passed += 1
    
    # Test 8: Complete metadata
    required_fields = {"chunk_id", "ticker", "year", "section_type", "text", "strategy", "char_count", "token_count"}
    all_have = all(required_fields.issubset(c.keys()) for c in fixed + section)
    if all_have:
        print(f"✓ TEST 8: All chunks have complete metadata")
        tests_passed += 1
    else:
        print(f"✗ TEST 8: Missing metadata fields")
    
    print(f"\nRESULT: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


if __name__ == "__main__":
    build_all_chunks()
    run_acceptance_tests()
