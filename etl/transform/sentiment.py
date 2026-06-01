from transformers import pipeline
import pandas as pd
from pathlib import Path


HAWKISH = ["inflation", "price pressure", "overheat", "restrictive", "above target", 
           "tight labor", "wage growth", "rate hike", "tighten"]
DOVISH  = ["slowdown", "weakness", "uncertainty", "easing", "below target", 
           "contraction", "layoffs", "slack", "recession", "decelerat"]


def hawk_dove_score(text: str) -> float:
    text_lower = text.lower()
    hawk = sum(1 for word in HAWKISH if word in text_lower)
    dove = sum(1 for word in DOVISH if word in text_lower)
    total = hawk + dove
    if total == 0:
        return 0.0
    return (hawk - dove) / total  # +1 = fully hawkish, -1 = fully dovish


def load_finbert():
    return pipeline(
        "text-classification",
        model = "ProsusAI/finbert",
        top_k = None
    )


def score_chunk(text: str, pipe) -> dict:
    result = pipe(text, truncation = True, max_length = 512)[0]
    scores = {item["label"]: item["score"] for item in result}
    return {
        "positive": scores.get("positive", 0),
        "negative": scores.get("negative", 0),
        "neutral": scores.get("neutral", 0)
    }


def score_document(doc_chunks: pd.DataFrame, pipe) -> dict:
    summary_chunks = doc_chunks[doc_chunks["section"] == "summary"]
    region_chunks = doc_chunks[doc_chunks["section"] != "summary"]

    def mean_scores(chunks):
        scores = [score_chunk(row["text"], pipe) for _, row in chunks.iterrows()]
        if not scores:
            return {"positive": 0, "negative": 0, "neutral": 0}
        return {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}

    summary_scores = mean_scores(summary_chunks)
    region_scores = mean_scores(region_chunks)
    all_text = " ".join(doc_chunks["text"].tolist())

    return {
        "date": doc_chunks["date"].iloc[0],
        "summary_positive": summary_scores["positive"],
        "summary_negative": summary_scores["negative"],
        "summary_neutral": summary_scores["neutral"],
        "region_positive": region_scores["positive"],
        "region_negative": region_scores["negative"],
        "region_neutral": region_scores["neutral"],
        "hawk_dove_score": hawk_dove_score(all_text),
        "summary_hawk_dove": hawk_dove_score(" ".join(summary_chunks["text"].tolist()))
    }    


def hawk_dove_score(text: str) -> float:
    text_lower = text.lower()
    hawk = sum(1 for word in HAWKISH if word in text_lower)
    dove = sum(1 for word in DOVISH if word in text_lower)
    total = hawk + dove
    if total == 0:
        return 0.0
    return (hawk - dove) / total


def score_all_documents(chunks_df: pd.DataFrame, pipe) -> pd.DataFrame:
    results = []
    for date, group in chunks_df.groupby("date"):
        print(f"Scoring {date}...")
        results.append(score_document(group,pipe))
    return pd.DataFrame(results)

if __name__ == "__main__":
    from pathlib import Path
    chunks_df = pd.read_parquet(Path(__file__).parent.parent.parent / "data" / "cleaned_chunks.parquet")
    pipe = load_finbert()
    sentiment_df = score_all_documents(chunks_df, pipe)
    output_path = Path(__file__).parent.parent.parent / "data" / "sentiment_features.parquet"
    sentiment_df.to_parquet(output_path, index=False)
    print(f"Saved {len(sentiment_df)} rows to {output_path}")
