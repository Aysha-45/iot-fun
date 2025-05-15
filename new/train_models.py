#!/usr/bin/env python
"""
Train two SVM pipelines with StandardScaler:
  • temp_SVM.pkl   – predicts temperature from scaled [time_processed, humid]
  • humid_SVM.pkl  – predicts humidity   from scaled [time_processed, temp]
Each .pkl stores {pipeline:<Pipeline>, features:<list[str]>}.
"""

import os, joblib, pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# ───────── config ─────────
DATA_CSV  = "./feed.csv"
MODEL_DIR = "./models"
TEST_SIZE = 0.2
SEED      = 42
SVR_KW    = dict(kernel="rbf", C=100, gamma="scale")

os.makedirs(MODEL_DIR, exist_ok=True)

# ───────── load & prep ─────────
df = pd.read_csv(DATA_CSV)
df["time"] = pd.to_datetime(df["time"], dayfirst=True, format="%d/%m/%Y %H:%M")
df["time_processed"] = df["time"].astype("int64") // 10**9

def fit_pipeline(X, y, fname):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svm",   SVR(**SVR_KW))
    ]).fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    print(f"[{fname}] MSE={mean_squared_error(y_te,y_pred):.2f} "
          f"R²={r2_score(y_te,y_pred):.2f}")

    joblib.dump({"pipeline": pipe, "features": list(X.columns)},
                os.path.join(MODEL_DIR, fname))

# temp model: time+humid → temp
fit_pipeline(df[["time_processed", "humid"]], df["temp"],  "temp_SVM.pkl")

# humid model: time+temp → humid
fit_pipeline(df[["time_processed", "temp"]],  df["humid"], "humid_SVM.pkl")

print("✓ Training finished", datetime.now().isoformat(timespec="seconds"))
