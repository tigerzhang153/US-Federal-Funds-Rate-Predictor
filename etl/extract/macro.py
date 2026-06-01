import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

unrate = fred.get_series("UNRATE").reset_index()
unrate.columns = ["date", "unemployment"]
unrate["date"] = pd.to_datetime(unrate["date"])
time.sleep(1)

cpi = fred.get_series("CPIAUCSL").reset_index()
cpi.columns = ["date", "cpi"]
cpi["date"] = pd.to_datetime(cpi["date"])
cpi["cpi_change"] = cpi["cpi"].pct_change() * 100  # month-over-month % change
cpi = cpi[["date", "cpi_change"]]
time.sleep(1)

rate = fred.get_series("DFEDTARU").reset_index()
rate.columns = ["date", "rate_level"]
rate["date"] = pd.to_datetime(rate["date"])

macro = unrate.merge(cpi, on="date", how="outer").merge(rate, on="date", how="outer")
macro = macro.sort_values("date")

# load beige book dates
bb_dates = pd.read_parquet(Path(__file__).parent.parent.parent / "data" / "sentiment_features.parquet")[["date"]]
bb_dates = bb_dates.sort_values("date")

# for each beige book date, get the most recent macro values available before that date
aligned = pd.merge_asof(bb_dates, macro, on="date", direction="backward")
print(aligned.head())

output_path = Path(__file__).parent.parent.parent / "data" / "macro_features.parquet"
aligned.to_parquet(output_path, index=False)
print(f"Saved {len(aligned)} rows")


print(unrate.tail(5))
