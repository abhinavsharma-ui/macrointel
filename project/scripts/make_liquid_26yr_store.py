from pathlib import Path
import os, shutil, math
import pandas as pd
import numpy as np

SRC = Path(os.getenv("FEATURE_26YR_SRC", "data/features_26yr"))
OUT = Path(os.getenv("FEATURE_26YR_LIQUID_OUT", "data/features_26yr_liquid"))
MIN_ROWS = int(os.getenv("LIQUID_MIN_ROWS", "1250"))
MIN_DOLLAR_VOL = float(os.getenv("LIQUID_MIN_DOLLAR_VOL", "20000000"))
MAX_SYMBOLS = int(os.getenv("LIQUID_MAX_SYMBOLS", "1200"))

OUT.mkdir(parents=True, exist_ok=True)
for x in OUT.glob("*.parquet"):
    x.unlink()

rows = []
for f in sorted(SRC.glob("*.parquet")):
    sym = f.stem.upper()
    if ".NS" in sym or "USDT" in sym or "USD" in sym or "-" in sym:
        continue
    try:
        df = pd.read_parquet(f, columns=["close","volume"])
        if len(df) < MIN_ROWS:
            continue
        idx = pd.to_datetime(df.index, errors="coerce", utc=True)
        if idx.max() < pd.Timestamp("2025-01-01", tz="UTC"):
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        recent = pd.DataFrame({"close": close, "volume": vol}).dropna().tail(252)
        if recent.empty:
            continue
        last_close = float(recent["close"].iloc[-1])
        med_dollar = float((recent["close"] * recent["volume"]).median())
        if last_close < 5 or med_dollar < MIN_DOLLAR_VOL:
            continue
        years = max(0.1, (idx.max() - idx.min()).days / 365.25)
        score = math.log1p(med_dollar) + min(years, 20) * 0.15 + min(len(df), 6500) / 6500
        rows.append((score, sym, str(f), len(df), str(idx.min()), str(idx.max()), last_close, med_dollar))
    except Exception:
        pass

rows.sort(reverse=True)
rows = rows[:MAX_SYMBOLS]

for _, sym, src, *_ in rows:
    dst = OUT / f"{sym}.parquet"
    try:
        os.symlink(Path(src).resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)

report = OUT / "_liquid_report.csv"
pd.DataFrame(rows, columns=["score","symbol","path","rows","min_date","max_date","last_close","median_dollar_vol"]).to_csv(report, index=False)
print("liquid_symbols", len(rows))
print("out", OUT)
print("report", report)
print(pd.read_csv(report).head(25).to_string(index=False))
