# Fed Rate Decision Prediction via NLP

Predicting Federal Reserve rate cut decisions ahead of FOMC meetings using NLP on Beige Book text combined with macroeconomic features.

---

## Project Overview

This project builds an end-to-end pipeline that scrapes Federal Reserve Beige Books, runs NLP to extract sentiment and narrative signals, combines them with macro data, and predicts whether the Fed will cut rates at the next FOMC meeting.

The pipeline is designed for continuous operation — re-running it after each new Beige Book publication automatically incorporates the latest data.

---

## Architecture

```
Web Scrape (federalreserve.gov)
│
├── etl/extract/fed.py       → Beige Book text + regional sections
├── etl/extract/fomc.py      → FOMC rate decisions (FRED API)
└── etl/extract/macro.py     → Unemployment, CPI, rate level (FRED API)
│
▼
etl/transform/clean.py       → Section cleaning + FinBERT chunking
etl/transform/sentiment.py   → FinBERT scoring + hawkish/dovish lexicon
etl/transform/embeddings.py  → Narrative shift via sentence embeddings
│
▼
etl/load/store.py            → Time-aligned feature matrix with FOMC labels
│
▼
models/train.py              → XGBoost classifier with time series CV
models/predict.py            → Score latest Beige Book → P(cut) / P(no_cut)
```

---

## Features

**NLP features (per Beige Book):**
- FinBERT positive/negative/neutral scores for national summary and regional sections
- Hawkish/dovish keyword lexicon score
- Narrative shift — cosine distance between consecutive Beige Book embeddings

**Macro features (FRED, backward-looking join):**
- Unemployment rate (`UNRATE`)
- CPI month-over-month change (`CPIAUCSL`)
- Current fed funds target rate (`DFEDTARU`)

**Label:** binary — `cut` vs `no_cut` at the next FOMC meeting, assigned via forward-looking join with a 45-day tolerance.

---

## Technologies

- Python 3.13
- `requests` / `beautifulsoup4` — web scraping
- `fredapi` — FRED macroeconomic data
- `nltk` — sentence tokenization
- `transformers` — FinBERT (`ProsusAI/finbert`)
- `sentence-transformers` — narrative shift (`all-MiniLM-L6-v2`)
- `scikit-learn` — preprocessing, time series CV
- `xgboost` — classifier
- `pandas` / `pyarrow` — data pipeline and Parquet storage

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the project root:
```
FRED_API_KEY=your_key_here
```

Get a free API key at [fred.stlouisfed.org](https://fred.stlouisfed.org).

---

## Running the Pipeline

```bash
# Extract
python etl/extract/fed.py
python etl/extract/fomc.py
python etl/extract/macro.py

# Transform
python etl/transform/clean.py
python etl/transform/sentiment.py
python etl/transform/embeddings.py

# Load
python etl/load/store.py

# Train and predict
python models/train.py
python models/predict.py
```

---

## Data

- **Source:** Federal Reserve Beige Book summary pages (2017–present)
- **Coverage:** ~75 documents, published ~8 times per year
- **Storage:** Parquet files in `data/`

---

## Limitations

- Small dataset (~75 samples, ~8 cut examples) limits cut recall
- Prediction quality will improve as more rate cut cycles are observed
- Model is retrained on the full dataset before each prediction
