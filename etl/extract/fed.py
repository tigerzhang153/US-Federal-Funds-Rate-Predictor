import requests
from datetime import datetime, date
import calendar
from bs4 import BeautifulSoup
import re
import pandas as pd
from pathlib import Path

ARCHIVE_URL = "https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm"
BASE_URL = "https://www.federalreserve.gov"

DISTRICTS = ["Boston", "New York", "Philadelphia", "Cleveland", 
             "Richmond", "Atlanta", "Chicago", "St. Louis", 
             "Minneapolis", "Kansas City", "Dallas", "San Francisco"]


def get_all_urls(since: datetime = None):
    if since is None:
        since = datetime(2017, 1, 1)
    
    urls = []
    current = since.replace(day=1)
    today = datetime.today()
    
    while current <= today:
        yyyymm = current.strftime("%Y%m")
        url = f"{BASE_URL}/monetarypolicy/beigebook{yyyymm}-summary.htm"
        response = requests.head(url)
        if response.status_code == 200:
            urls.append(url)
        # advance one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return urls


def extract_regions(text: str) -> dict:
    positions = []
    for district in DISTRICTS:
        idx = text.find(district)
        if idx != -1:
            positions.append((idx, district))
    positions.sort(key=lambda x: x[0])
    
    regions = {}
    for i, (start, district) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(text)
        regions[district] = text[start:end].strip()
    return regions


def parse_beige_book(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, "html.parser")

    # Date from URL
    m = re.search(r"(\d{6})", url)
    date = datetime.strptime(m.group(1), "%Y%m") if m else None
    
    # Walk Tags grouping <p> under headings
    body = soup.find("article") or soup.body or soup
    sections = {}
    current = "_preamble"
    sections[current] = []

    for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        if tag.name.startswith("h"):
            current = tag.get_text(strip = True)
            sections.setdefault(current, [])
        else:
            text = tag.get_text(" ", strip = True)
            if text:
                sections.setdefault(current, []).append(text)

    # Build Outputs 
    section_text = {h: "\n".join(ps) for h, ps in sections.items() if ps}


    #Summary: heading whose name looks like a summary, else first section
    summary = ""
    for h, t in section_text.items():
        if "summary" in h.lower() or "overall" in h.lower():
            summary = t
            break
    
    if not summary:
        summary = next(iter(section_text.values()), "")
    
    full_text = summary

    # Regions: every real (non-summary, non-preamble) heading
    ns_text = section_text.get("National Summary", "")
    highlights_start = ns_text.find("Highlights by Federal Reserve District")
    highlights = ns_text[highlights_start:] if highlights_start != -1 else ns_text
    regions = extract_regions(highlights)

    return {
        "date": date,
        "full_text": full_text,
        "summary": summary,
        "regions": regions,
    }



urls = get_all_urls()
result = parse_beige_book(urls[0])

def scrape_all(since=None):
    urls = get_all_urls(since)
    results = []
    for url in urls:
        print(f"Scraping {url}...")
        results.append(parse_beige_book(url))
    df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent.parent / "data" / "extracted_beige_books.parquet"
    output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    scrape_all()





