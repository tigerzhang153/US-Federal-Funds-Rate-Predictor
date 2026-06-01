from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path


def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str, model) -> np.ndarray:
    return model.encode(text, normalize_embeddings = True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))        # vectors are already normalized so dot product = cosine similarity


def compute_embeddings(chunks_df: pd.DataFrame, model) -> pd.DataFrame:
    # get one summary text per document (join all summary chunks)
    summaries = (
        chunks_df[chunks_df["section"] == "summary"]
        .groupby("date")["text"]
        .apply(" ".join)
        .reset_index()
        .sort_values("date")
    )

    summaries["embedding"] = summaries["text"].apply(lambda t: embed_text(t, model))

    # narrative shift = 1 - cosine similarity with previous book
    embeddings = summaries["embedding"].tolist()
    shifts = [None]     # first document has no previous
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i], embeddings[i - 1])
        shifts.append(1 - sim)

    summaries["narrative_shift"] = shifts
    return summaries[["date", "embedding", "narrative_shift"]]


if __name__ == "__main__":
    chunks_df = pd.read_parquet(Path(__file__).parent.parent.parent / "data" / "cleaned_chunks.parquet")
    model = load_model()
    emb_df = compute_embeddings(chunks_df, model)
    output = emb_df[["date", "narrative_shift"]]
    output_path = Path(__file__).parent.parent.parent / "data" / "embedding_features.parquet"
    output.to_parquet(output_path, index=False)
    print(f"Saved {len(output)} rows")

