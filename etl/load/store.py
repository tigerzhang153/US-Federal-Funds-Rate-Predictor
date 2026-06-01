import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent.parent / "data"

sentiment_df = pd.read_parquet(data_dir / "sentiment_features.parquet")
embedding_df = pd.read_parquet(data_dir / "embedding_features.parquet")
fomc_df = pd.read_parquet(data_dir / "fomc_labels.parquet")

features = sentiment_df.merge(embedding_df, on = "date", how = "inner")

macro_df = pd.read_parquet(data_dir / "macro_features.parquet")
features = features.merge(macro_df, on="date", how="left")

features = features.sort_values("date")
fomc_df = fomc_df.sort_values("date")

features = pd.merge_asof(
    features,
    fomc_df[["date", "label"]],
    on = "date",
    direction = "forward",
    tolerance = pd.Timedelta("45 days")
)
features["label"] = features["label"].fillna("hold")
features["label"] = features["label"].apply(lambda x: "cut" if x == "cut" else "no_cut")


features = features.dropna(subset = ["narrative_shift"])

output_path = data_dir / "feature_matrix.parquet"
features.to_parquet(output_path, index = False)
print(features.shape)
print(features["label"].value_counts())