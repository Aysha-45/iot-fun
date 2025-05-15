#!/usr/bin/env python
"""
compare_hourly.py
─────────────────
Call /predict 10 times.  Each request’s timestamp is base_time + n·1 hour,
so the model sees a full-day span even though the script runs in ~10 s.

Usage
-----
python compare_hourly.py

Prereqs
-------
pip install requests pandas matplotlib
"""

import time, random, requests, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

API        = "http://127.0.0.1:5001/predict"
N_SAMPLES  = 10
GAP_HOURS  = 1                 # virtual time step
REAL_PAUSE = 1                 # seconds to wait between HTTP calls

# ────────── helper to fabricate live sensor readings ──────────
def get_measurement():
    """Replace with real sensor reads if available."""
    return {
        "temperature": round(30 + random.uniform(-2, 2), 1),
        "humidity":    round(70 + random.uniform(-5, 5), 1),
    }

# ────────── main loop ──────────
base_dt = datetime.now(timezone.utc)      # starting point in UTC
rows = []

print("Collecting data (virtual span = 9 hours)…")
for i in range(N_SAMPLES):
    dt  = base_dt + timedelta(hours=i * GAP_HOURS)
    iso = dt.isoformat()                  # e.g. "2025-05-14T13:00:00+00:00"

    point = get_measurement()
    payload = {**point, "timestamp": iso}

    r = requests.post(API, json=payload, timeout=5)
    r.raise_for_status()
    out = r.json()

    rows.append({
        "idx":           i + 1,
        "api_time":      out["timestamp"],
        "temp_real":     out["current"]["temperature"],
        "temp_pred":     out["predicted"]["temperature"],
        "humid_real":    out["current"]["humidity"],
        "humid_pred":    out["predicted"]["humidity"],
    })

    print(f"[{i+1:02}] {iso}  sent {point} → "
          f"T_pred={out['predicted']['temperature']:.2f}, "
          f"H_pred={out['predicted']['humidity']:.2f}")

    if i < N_SAMPLES - 1:
        time.sleep(REAL_PAUSE)

df = pd.DataFrame(rows)
print("\nSummary\n", df[["idx", "api_time", "temp_real", "temp_pred",
                        "humid_real", "humid_pred"]])

# ────────── plot ──────────
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(df["idx"], df["temp_real"],  marker="o", label="Actual")
ax[0].plot(df["idx"], df["temp_pred"], marker="x", label="Predicted")
ax[0].set_title("Temperature (°C)")
ax[0].set_xlabel("Sample (each = +1 h)")
ax[0].set_ylabel("°C")
ax[0].legend()

ax[1].plot(df["idx"], df["humid_real"], marker="o", label="Actual")
ax[1].plot(df["idx"], df["humid_pred"], marker="x", label="Predicted")
ax[1].set_title("Humidity (%)")
ax[1].set_xlabel("Sample (each = +1 h)")
ax[1].set_ylabel("%")
ax[1].legend()

fig.suptitle("Model predictions vs real measurements (virtual 10-hour span)")
plt.tight_layout()
plt.show()
