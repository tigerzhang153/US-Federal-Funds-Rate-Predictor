# Fed Rate Decision Prediction via NLP

Predicting Federal Reserve rate cuts ahead of policy meetings using NLP and structured feature engineering from unstructured macroeconomic text.

---

## Project Overview

This project builds a production-style, reproducible pipeline to forecast the probability of FOMC (Federal Open Market Committee) rate cuts by extracting insights from:

- FOMC statements, meeting minutes, and speeches  
- Press conference transcripts  
- Macro news articles  
- Investment bank research reports  

The pipeline converts raw text (including PDFs) into structured, time-indexed features using transformer-based NLP and sentence embeddings. Probabilistic models are trained on these features to predict rate-cut decisions before meetings, while controlling for look-ahead bias and class imbalance.

---

## Architecture
Raw Documents (PDF / Text)
│
▼
ETL: Extract → Transform → Load
│
▼
Feature Engineering:
Hawkish/Dovish sentiment
Inflation & growth concern
Sentence embeddings & narrative shifts
│
▼
Time-aligned Features → Model Training
│
▼
Predictions:
Probability of rate cut at next FOMC meeting

---

## Technologies Used

- Python 3.13 – Core programming
- pandas – Data manipulation
- pdfplumber / PyMuPDF / pytesseract – PDF & OCR extraction
- nltk / spaCy – Text preprocessing
- Hugging Face Transformers / sentence-transformers – NLP models for sentiment and embeddings
- scikit-learn / XGBoost – Baseline ML models
- Parquet / SQLite – Feature and metadata storage
- yaml – Configuration management

---

## Features

- ETL pipeline: Extract text from PDFs and raw text, transform (clean, chunk, normalize), and load into structured datasets.
- Policy-specific NLP: Quantify hawkish/dovish stance, inflation and growth concerns, narrative shifts.
- Embeddings: Compute semantic representations using transformer models, aggregated over time.
- Time-series alignment: Features are aligned to FOMC meetings to avoid look-ahead bias.
- Probabilistic prediction: Outputs probability of rate cut before each meeting, benchmarked against historical outcomes and market expectations.
