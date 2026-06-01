from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path


load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

series = fred.get_series("DFEDTARU")
df = series.reset_index()
df.columns = ["date", "rate"]
df["date"] = pd.to_datetime(df["date"])

df = df.dropna(subset=["rate"])
df["change_bps"] = df["rate"].diff() * 100   #convert percentage points to basis points
df["change_bps"] = df["change_bps"].round(2)


decisions = df[df["change_bps"] != 0].copy()
decisions["label"] = decisions["change_bps"].apply(lambda x: "cut" if x < 0 else "hike")

output_path = Path(__file__).parent.parent.parent / "data" / "fomc_labels.parquet"
output_path.parent.mkdir(exist_ok=True)
decisions.to_parquet(output_path, index=False)
