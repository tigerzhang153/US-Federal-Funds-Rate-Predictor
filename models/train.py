import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder



data_dir = Path(__file__).parent.parent / "data"
df = pd.read_parquet(data_dir / "feature_matrix.parquet")

feature_cols = [c for c in df.columns if c not in ["date", "label"]]
X = df[feature_cols]
y = df["label"]
y = y.map({"cut": 1, "no_cut": 0})


tscv = TimeSeriesSplit(n_splits = 3)
model = LogisticRegression(class_weight = "balanced", max_iter = 1000)

all_preds = []
all_true = []

"""
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test)

    all_preds.extend(preds)
    all_true.extend(y_test)

print(classification_report(all_true, all_preds))
"""


le = LabelEncoder()
y_encoded = le.fit_transform(y)

cut_count = (y == 1).sum()
no_cut_count = (y == 0).sum()
scale = no_cut_count / cut_count

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss",
    scale_pos_weight=scale
)


all_preds_xgb = []
all_true_xgb = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    xgb.fit(X_train_scaled, y_train)
    
    probs = xgb.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.3).astype(int)

    all_preds_xgb.extend(preds)
    all_true_xgb.extend(y_test.tolist())



for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    print(f"Fold {i}: train cuts={y_train.sum()}, test cuts={y_test.sum()}")


print(classification_report(all_true_xgb, all_preds_xgb))

import joblib

scaler_final = StandardScaler()
X_scaled_all = scaler_final.fit_transform(X)
xgb.fit(X_scaled_all, y)

model_dir = Path(__file__).parent
joblib.dump(xgb, model_dir / "xgb_model.pkl")
joblib.dump(scaler_final, model_dir / "scaler.pkl")
print("Model saved.")

