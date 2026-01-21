import pdfplumber
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from Beige Book filename like 'BeigeBook_20240117.pdf'"""
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text


def extract_regional_sections(text: str) -> Dict[str, str]:
    """
    Extract regional sections from Beige Book text.
    Common regional headers: First District--Boston, Second District--New York, etc.
    """
    regions = {}
    
    # Pattern to match regional headers (various formats)
    # Examples: "First District--Boston", "Second District—New York", "ATLANTA"
    region_patterns = [
        r'First District[—-]\s*Boston',
        r'Second District[—-]\s*New York',
        r'Third District[—-]\s*Philadelphia',
        r'Fourth District[—-]\s*Cleveland',
        r'Fifth District[—-]\s*Richmond',
        r'Sixth District[—-]\s*Atlanta',
        r'Seventh District[—-]\s*Chicago',
        r'Eighth District[—-]\s*St\.?\s*Louis',
        r'Ninth District[—-]\s*Minneapolis',
        r'Tenth District[—-]\s*Kansas City',
        r'Eleventh District[—-]\s*Dallas',
        r'Twelfth District[—-]\s*San Francisco',
        r'BOSTON|ATLANTA|CHICAGO|CLEVELAND|DALLAS|KANSAS CITY|MINNEAPOLIS|NEW YORK|PHILADELPHIA|RICHMOND|SAN FRANCISCO|ST\.?\s*LOUIS',
    ]
    
    # Find all region matches
    region_matches = []
    for pattern in region_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            region_matches.append((match.start(), match.group()))
    
    # Sort by position
    region_matches.sort(key=lambda x: x[0])
    
    # Extract text between regions
    for i in range(len(region_matches)):
        start_pos = region_matches[i][0]
        region_name = region_matches[i][1]
        
        # Find end position (next region or end of document)
        if i + 1 < len(region_matches):
            end_pos = region_matches[i + 1][0]
        else:
            end_pos = len(text)
        
        region_text = text[start_pos:end_pos].strip()
        regions[region_name] = region_text
    
    return regions


def extract_summary_section(text: str) -> Optional[str]:
    """Extract the summary section (usually at the beginning or end)."""
    # Look for common summary headers
    summary_patterns = [
        r'SUMMARY',
        r'National Summary',
        r'Overview',
    ]
    
    for pattern in summary_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Extract text after summary header, up to first region or end
            start_pos = match.end()
            region_match = re.search(r'First District|Second District|BOSTON|ATLANTA', text[start_pos:], re.IGNORECASE)
            if region_match:
                return text[start_pos:start_pos + region_match.start()].strip()
            return text[start_pos:].strip()
    
    return None


def clean_text(text: str) -> str:
    """Clean extracted text for NLP processing."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and headers/footers
    text = re.sub(r'Page \d+', '', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    return text.strip()


def extract_beige_book(pdf_path: Path) -> Dict:
    """
    Main extraction function for a Beige Book PDF.
    Returns a dictionary with extracted information.
    """
    date = extract_date_from_filename(pdf_path.name)
    
    # Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return {"error": f"Could not extract text from {pdf_path.name}"}
    
    # Extract structured sections
    summary = extract_summary_section(raw_text)
    regions = extract_regional_sections(raw_text)
    
    # Clean text
    cleaned_full_text = clean_text(raw_text)
    cleaned_summary = clean_text(summary) if summary else None
    
    return {
        "filename": pdf_path.name,
        "date": date,
        "full_text": cleaned_full_text,
        "summary": cleaned_summary,
        "regions": {k: clean_text(v) for k, v in regions.items()},
        "region_count": len(regions),
        "char_count": len(cleaned_full_text),
        "word_count": len(cleaned_full_text.split()),
    }


def process_all_beige_books(raw_data_dir: Path) -> pd.DataFrame:
    """
    Process all Beige Book PDFs in the raw_data directory.
    Returns a DataFrame with extracted information.
    """
    beige_book_dir = raw_data_dir / "Fed Beige Book"
    
    if not beige_book_dir.exists():
        raise FileNotFoundError(f"Directory not found: {beige_book_dir}")
    
    results = []
    
    # Process each PDF
    pdf_files = sorted(beige_book_dir.glob("BeigeBook_*.pdf"))
    print(f"Found {len(pdf_files)} Beige Book PDFs")
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        extracted = extract_beige_book(pdf_path)
        results.append(extracted)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Expand regions into separate rows if needed (optional)
    # For now, keep regions as dict in each row
    
    return df


if __name__ == "__main__":
    # Path to raw data directory
    raw_data_path = Path(__file__).parent.parent / "raw_data"
    
    # Extract all Beige Books
    df = process_all_beige_books(raw_data_path)
    
    # Save extracted data
    output_path = Path(__file__).parent.parent.parent / "data" / "extracted_beige_books.parquet"
    output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\nExtraction complete!")
    print(f"Processed {len(df)} documents")
    print(f"Total words: {df['word_count'].sum():,}")
    print(f"Saved to: {output_path}")