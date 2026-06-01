import joblib
import pandas as pd
from pathlib import Path

model_dir = Path(__file__).parent
data_dir = model_dir.parent / "data"

xgb = joblib.load(model_dir / "xgb_model.pkl")
scaler = joblib.load(model_dir/ "scaler.pkl")

feature_matrix = pd.read_parquet(data_dir / "feature_matrix.parquet")
feature_cols = [c for c in feature_matrix.columns if c not in ["date", "label"]]
latest = feature_matrix.sort_values("date").iloc[[-1]]
X_latest = latest[feature_cols]
X_scaled = scaler.transform(X_latest)


probs = xgb.predict_proba(X_scaled)[0]
print(f"Beige Book date: {latest['date'].iloc[0].date()}")
print(f"P(cut):    {probs[1]:.1%}")
print(f"P(no_cut): {probs[0]:.1%}")
