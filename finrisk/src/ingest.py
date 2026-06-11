"""
FinRisk — Stage 1: SEC Filing Ingestion

Downloads 10-K filings from SEC EDGAR and parses them to extract:
  - Item 1A (Risk Factors)
  - Item 7  (MD&A)
  - Item 8  (Financial Statements)

Saves extracted sections as data/processed/finrisk_sections.parquet
"""

import re
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    COMPANIES, YEARS, RAW_DIR, PROCESSED_DIR,
    SEC_COMPANY_NAME, SEC_EMAIL, DATA_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Section extraction patterns
# ──────────────────────────────────────────────────────────────

# Regex patterns to locate Item boundaries in 10-K filings.
# These handle both plain-text and HTML-embedded text.
# Pattern strategy: find the start of each Item, then capture text until the next Item.

ITEM_PATTERNS = {
    "Risk Factors": [
        # Pattern 1: "Item 1A" with optional period/dash, various separators
        r'(?:item\s*1a[\.\:\-—]?\s*(?:risk\s+factors))',
        # Pattern 2: just "ITEM 1A" as a standalone header
        r'(?:ITEM\s+1A[\.\:\-—]?\s*(?:RISK\s+FACTORS)?)',
    ],
    "MD&A": [
        r'(?:item\s*7[\.\:\-—]?\s*(?:management[\'\u2019]?s?\s+discussion\s+and\s+analysis))',
        r'(?:ITEM\s+7[\.\:\-—]?\s*(?:MANAGEMENT[\'\u2019]?S?\s+DISCUSSION\s+AND\s+ANALYSIS)?)',
    ],
    "Financial Statements": [
        r'(?:item\s*8[\.\:\-—]?\s*(?:financial\s+statements?\s+and\s+supplementary\s+data))',
        r'(?:ITEM\s+8[\.\:\-—]?\s*(?:FINANCIAL\s+STATEMENTS?\s+AND\s+SUPPLEMENTARY\s+DATA)?)',
    ],
}

# Patterns marking the END of a section (next Item header)
NEXT_ITEM_PATTERNS = {
    "Risk Factors": [  # Item 1A ends at Item 1B or Item 2
        r'(?:item\s*1b[\.\:\-—])',
        r'(?:ITEM\s+1B[\.\:\-—]?)',
        r'(?:item\s*2[\.\:\-—])',
        r'(?:ITEM\s+2[\.\:\-—]?)',
    ],
    "MD&A": [  # Item 7 ends at Item 7A or Item 8
        r'(?:item\s*7a[\.\:\-—])',
        r'(?:ITEM\s+7A[\.\:\-—]?)',
        r'(?:item\s*8[\.\:\-—])',
        r'(?:ITEM\s+8[\.\:\-—]?)',
    ],
    "Financial Statements": [  # Item 8 ends at Item 9
        r'(?:item\s*9[\.\:\-—])',
        r'(?:ITEM\s+9[\.\:\-—]?)',
    ],
}


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities using BeautifulSoup."""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    """Clean whitespace, remove excessive newlines, normalize spaces."""
    # Replace multiple whitespace/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any remaining HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Clean up
    text = text.strip()
    return text


def extract_section(filing_text: str, section_type: str) -> Optional[str]:
    """
    Extract a specific section from a 10-K filing text.
    
    Strategy:
    1. Find all occurrences of the section header
    2. For each occurrence, find the next Item header (section end)
    3. Take the longest extraction (the real content, not TOC references)
    4. Clean HTML and whitespace
    """
    results = []
    text_lower = filing_text.lower()
    
    # Find all start positions using start patterns
    start_positions = []
    for pattern in ITEM_PATTERNS[section_type]:
        for m in re.finditer(pattern, filing_text, re.IGNORECASE):
            start_positions.append(m.start())
    
    if not start_positions:
        return None
    
    # For each start position, find the nearest end position
    for start_pos in start_positions:
        end_positions = []
        for pattern in NEXT_ITEM_PATTERNS[section_type]:
            for m in re.finditer(pattern, filing_text[start_pos + 50:], re.IGNORECASE):
                end_positions.append(start_pos + 50 + m.start())
        
        if end_positions:
            # Take the nearest end after start
            end_pos = min(end_positions)
        else:
            # No end found — take next 200K chars as fallback
            end_pos = start_pos + 200_000
        
        raw_section = filing_text[start_pos:end_pos]
        
        # Strip HTML
        section_text = strip_html(raw_section)
        # Clean whitespace
        section_text = clean_text(section_text)
        
        results.append(section_text)
    
    if not results:
        return None
    
    # Return the longest extraction (likely the real content, not TOC entry)
    best = max(results, key=len)
    return best if len(best) >= 500 else None


def extract_year_from_accession(accession: str) -> Optional[int]:
    """
    Extract filing year from accession number.
    Format: 0000320193-24-095896 → 24 → 2024
    """
    parts = accession.split("-")
    if len(parts) >= 2:
        try:
            yr_short = int(parts[1])
            year = 2000 + yr_short if yr_short < 50 else 1900 + yr_short
            return year
        except ValueError:
            pass
    return None


def extract_year_from_filing(filing_text: str, accession: str) -> int:
    """
    Try to extract the fiscal year from the filing content or accession number.
    Falls back to accession-based year if content parsing fails.
    """
    # Try accession number first
    year = extract_year_from_accession(accession)
    if year and year in range(2020, 2030):
        return year
    
    # Try to find "FISCAL YEAR ENDED" or "FOR THE YEAR ENDED" patterns
    patterns = [
        r'(?:fiscal\s+year\s+ended|for\s+the\s+year\s+ended|year\s+ended)\s+\w+\s+\d{1,2},?\s+(20\d{2})',
        r'(?:FISCAL\s+YEAR\s+ENDED|FOR\s+THE\s+YEAR\s+ENDED|YEAR\s+ENDED)\s+\w+\s+\d{1,2},?\s+(20\d{2})',
    ]
    for pattern in patterns:
        m = re.search(pattern, filing_text[:10000])
        if m:
            return int(m.group(1))
    
    # Fallback: use accession year - 1 (filing year is usually year before filing date)
    if year:
        return year - 1
    return 0


def download_filings():
    """
    Download 10-K filings for all companies using sec-edgar-downloader.
    Downloads into DATA_DIR / 'sec-edgar-filings'.
    """
    download_dir = str(DATA_DIR)
    dl = Downloader(SEC_COMPANY_NAME, SEC_EMAIL, download_dir)
    
    total = len(COMPANIES)
    for i, ticker in enumerate(COMPANIES, 1):
        # Check if already downloaded
        ticker_dir = DATA_DIR / "sec-edgar-filings" / ticker
        if ticker_dir.exists():
            existing = list(ticker_dir.rglob("full-submission.txt"))
            if len(existing) >= 3:
                logger.info(f"[{i}/{total}] {ticker}: Already have {len(existing)} filings, skipping")
                continue
        
        logger.info(f"[{i}/{total}] Downloading 10-K for {ticker}...")
        try:
            dl.get("10-K", ticker, limit=3)
            logger.info(f"[{i}/{total}] {ticker}: Download complete")
        except Exception as e:
            logger.error(f"[{i}/{total}] {ticker}: Download failed — {e}")


def copy_existing_filings():
    """
    Copy already-downloaded MSFT filings from the parent data directory
    to finrisk/data/sec-edgar-filings/ if they exist.
    """
    import shutil
    
    src = Path(__file__).parent.parent.parent / "data" / "sec-edgar-filings"
    dst = DATA_DIR / "sec-edgar-filings"
    
    if src.exists() and not (dst / "MSFT").exists():
        logger.info(f"Copying existing filings from {src} to {dst}")
        if src.is_dir():
            for ticker_dir in src.iterdir():
                if ticker_dir.is_dir() and ticker_dir.name != ".DS_Store":
                    dst_ticker = dst / ticker_dir.name
                    if not dst_ticker.exists():
                        shutil.copytree(ticker_dir, dst_ticker)
                        logger.info(f"  Copied {ticker_dir.name}")


def parse_all_filings() -> pd.DataFrame:
    """
    Parse all downloaded filings and extract sections.
    Returns DataFrame with columns: ticker, year, section_type, raw_text
    """
    filings_dir = DATA_DIR / "sec-edgar-filings"
    
    if not filings_dir.exists():
        logger.error(f"No filings directory found at {filings_dir}")
        return pd.DataFrame()
    
    rows = []
    filing_files = list(filings_dir.rglob("full-submission.txt"))
    logger.info(f"Found {len(filing_files)} filing files to parse")
    
    for idx, fpath in enumerate(filing_files, 1):
        # Extract ticker from path: .../sec-edgar-filings/TICKER/10-K/accession/full-submission.txt
        parts = fpath.parts
        try:
            # Find 'sec-edgar-filings' in path, ticker is next
            sef_idx = parts.index("sec-edgar-filings")
            ticker = parts[sef_idx + 1]
        except (ValueError, IndexError):
            logger.warning(f"Cannot determine ticker from path: {fpath}")
            continue
        
        # Get accession number from parent directory name
        accession = fpath.parent.name
        
        logger.info(f"[{idx}/{len(filing_files)}] Parsing {ticker} / {accession}...")
        
        # Read filing
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"  Cannot read file: {e}")
            continue
        
        # Extract year
        year = extract_year_from_filing(text, accession)
        if year == 0:
            logger.warning(f"  Cannot determine year for {ticker}/{accession}")
            continue
        
        # Extract each section
        for section_type in ["Risk Factors", "MD&A", "Financial Statements"]:
            section_text = extract_section(text, section_type)
            if section_text and len(section_text) >= 500:
                rows.append({
                    "ticker": ticker,
                    "year": year,
                    "section_type": section_type,
                    "raw_text": section_text,
                })
                logger.info(f"  ✓ {section_type}: {len(section_text):,} chars")
            else:
                length = len(section_text) if section_text else 0
                logger.warning(f"  ✗ {section_type}: too short ({length} chars) or not found")
    
    df = pd.DataFrame(rows)
    return df


def build_sections_parquet():
    """
    Full pipeline: copy existing → download → parse → save parquet.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: SEC FILING INGESTION")
    logger.info("=" * 60)
    
    # Step 1: Copy existing filings from parent project
    copy_existing_filings()
    
    # Step 2: Download remaining filings
    download_filings()
    
    # Step 3: Parse all filings
    df = parse_all_filings()
    
    if df.empty:
        logger.error("No sections extracted! Check filing download and parsing.")
        return df
    
    # Step 4: Save to parquet
    output_path = PROCESSED_DIR / "finrisk_sections.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved {len(df)} sections to {output_path}")
    
    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("STAGE 1 — INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total rows:        {len(df)}")
    print(f"Unique tickers:    {df['ticker'].nunique()}")
    print(f"Year range:        {sorted(df['year'].unique())}")
    print(f"\nSection type distribution:")
    print(df['section_type'].value_counts().to_string())
    print(f"\nRows per ticker (avg): {len(df) / df['ticker'].nunique():.1f}")
    print(f"Min text length:   {df['raw_text'].str.len().min():,} chars")
    print(f"Max text length:   {df['raw_text'].str.len().max():,} chars")
    print(f"Avg text length:   {df['raw_text'].str.len().mean():,.0f} chars")
    
    return df


def run_acceptance_tests(df: pd.DataFrame = None):
    """Run Stage 1 acceptance tests."""
    print("\n" + "=" * 60)
    print("STAGE 1 — ACCEPTANCE TESTS")
    print("=" * 60)
    
    output_path = PROCESSED_DIR / "finrisk_sections.parquet"
    
    if df is None:
        if output_path.exists():
            df = pd.read_parquet(output_path)
        else:
            print("FAIL: Parquet file does not exist")
            return False
    
    tests_passed = 0
    tests_total = 7
    
    # Test 1: Parquet file exists
    if output_path.exists():
        print("✓ TEST 1: Parquet file exists")
        tests_passed += 1
    else:
        print("✗ TEST 1: Parquet file does not exist")
    
    # Test 2: Correct columns
    expected_cols = {"ticker", "year", "section_type", "raw_text"}
    if set(df.columns) >= expected_cols:
        print(f"✓ TEST 2: Columns correct — {list(df.columns)}")
        tests_passed += 1
    else:
        print(f"✗ TEST 2: Missing columns — have {list(df.columns)}, need {expected_cols}")
    
    # Test 3: At least 20 unique tickers
    n_tickers = df['ticker'].nunique()
    if n_tickers >= 20:
        print(f"✓ TEST 3: {n_tickers} unique tickers (≥20)")
        tests_passed += 1
    else:
        print(f"✗ TEST 3: Only {n_tickers} unique tickers (need ≥20)")
    
    # Test 4: At least 2 years per ticker on average
    years_per_ticker = df.groupby('ticker')['year'].nunique().mean()
    if years_per_ticker >= 2.0:
        print(f"✓ TEST 4: {years_per_ticker:.1f} years per ticker on average (≥2)")
        tests_passed += 1
    else:
        print(f"✗ TEST 4: Only {years_per_ticker:.1f} years per ticker average (need ≥2)")
    
    # Test 5: Three section types present
    section_types = set(df['section_type'].unique())
    expected_sections = {"Risk Factors", "MD&A", "Financial Statements"}
    if expected_sections.issubset(section_types):
        print(f"✓ TEST 5: All 3 section types present")
        tests_passed += 1
    else:
        print(f"✗ TEST 5: Missing sections — have {section_types}, need {expected_sections}")
    
    # Test 6: No row has raw_text shorter than 500 characters
    short_rows = df[df['raw_text'].str.len() < 500]
    if len(short_rows) == 0:
        print(f"✓ TEST 6: No rows under 500 chars")
        tests_passed += 1
    else:
        print(f"✗ TEST 6: {len(short_rows)} rows under 500 chars")
    
    # Test 7: Print summary stats
    print(f"✓ TEST 7: Summary stats printed above")
    tests_passed += 1
    
    print(f"\nRESULT: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


if __name__ == "__main__":
    df = build_sections_parquet()
    if not df.empty:
        run_acceptance_tests(df)
