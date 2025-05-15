#!/usr/bin/env python
"""
Flask API with optional timestamp field.
Works with models trained via the new scaled pipelines.
"""

import os, time, logging, joblib, pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify

# ───────── logging ─────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ───────── load models ─────────
MODEL_DIR = "./models"
temp_bundle  = joblib.load(os.path.join(MODEL_DIR, "temp_SVM.pkl"))
humid_bundle = joblib.load(os.path.join(MODEL_DIR, "humid_SVM.pkl"))

temp_pipe,  temp_feats  = temp_bundle["pipeline"], temp_bundle["features"]
humid_pipe, humid_feats = humid_bundle["pipeline"], humid_bundle["features"]

log.info("✓ Pipelines loaded")

# ───────── util ─────────
def to_epoch(ts):
    if isinstance(ts, (int, float)):
        return int(ts)
    dt = datetime.fromisoformat(str(ts))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

# ───────── Flask ─────────
app = Flask(__name__)
application = app

@app.route("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.route("/predict", methods=["POST"])
def predict():
    js = request.get_json(force=True)
    if not all(k in js for k in ("temperature", "humidity")):
        return {"error": "temperature and humidity required"}, 400

    t, h = float(js["temperature"]), float(js["humidity"])
    epoch = to_epoch(js.get("timestamp", time.time()))

    temp_df  = pd.DataFrame([[epoch, h]], columns=temp_feats)
    humid_df = pd.DataFrame([[epoch, t]], columns=humid_feats)

    t_pred = float(temp_pipe .predict(temp_df )[0])
    h_pred = float(humid_pipe.predict(humid_df)[0])

    return jsonify({
        "current":   {"temperature": t, "humidity": h},
        "predicted": {"temperature": round(t_pred,2), "humidity": round(h_pred,2)},
        "changes":   {"temperature": round(t_pred - t,2),
                      "humidity":    round(h_pred - h,2)},
        "timestamp": datetime.utcfromtimestamp(epoch).isoformat() + "Z"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
