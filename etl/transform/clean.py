import pandas as pd
from pathlib import Path
import re
import nltk

data_dir = Path(__file__).parent.parent.parent / "data"
df = pd.read_parquet(data_dir / "extracted_beige_books.parquet")

def clean_section(text: str, district_name: str = None) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    # strip district name from start
    if district_name and text.startswith(district_name):
        text = text[len(district_name):].strip()

    # replace non-breaking spaces
    text = text.replace("\xa0", " ")

    # collapse white space and newlines
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def chunk_text(text: str, max_tokens: int = 400) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current and current_len + n > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = n
        else:
            current.append(sentence)
            current_len += n
    
    if current:
        chunks.append(" ". join(current))
    
    return chunks


def chunk_document(row: pd.Series) -> list[dict]:
    chunks = []

    # Chunk the summary
    for i, text in enumerate(chunk_text(clean_section(row["summary"]))):
        chunks.append({
            "date": row["date"],
            "section": "summary",
            "chunk_index": i,
            "text": text
        })
    
    # Chunk each region
    for district, raw_text in row["regions"].items():
        for i, text in enumerate(chunk_text(clean_section(raw_text, district_name = district))):
            chunks.append({
                "date": row["date"],
                "section": district,
                "chunk_index": i,
                "text": text
            })
    
    return chunks


all_chunks = []
for _, row in df.iterrows():
    all_chunks.extend(chunk_document(row))

chunks_df = pd.DataFrame(all_chunks)
output_path = data_dir / "cleaned_chunks.parquet"
chunks_df.to_parquet(output_path, index=False)
print(f"Saved {len(chunks_df)} chunks from {len(df)} documents")
