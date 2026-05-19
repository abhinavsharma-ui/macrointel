import os, re, time, json, sqlite3
from pathlib import Path
import requests
import pandas as pd

API=os.environ["EODHD_API_KEY"]
OUT=Path(os.getenv("EODHD_PRICE_OUT","data/prices_survivorship_free"))
REPORT=Path("reports/eodhd_survivorship_import.json")
EXCHANGE=os.getenv("EODHD_EXCHANGE","US")
START=os.getenv("EODHD_START","2025-05-01")
MAX_SYMBOLS=int(os.getenv("EODHD_MAX_SYMBOLS","5"))
SLEEP=float(os.getenv("EODHD_SLEEP","0.5"))
INDEX_NAME=os.getenv("EODHD_INDEX_NAME","EODHD_US_ACTIVE_AND_DELISTED")

OUT.mkdir(parents=True, exist_ok=True)

def get_json(url, params):
    r=requests.get(url, params=params, timeout=45)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    return r.json()

def ticker_list(delisted):
    url=f"https://eodhd.com/api/exchange-symbol-list/{EXCHANGE}"
    data=get_json(url, {"api_token":API, "fmt":"json", "delisted":int(delisted)})
    rows=[]
    for x in data:
        code=str(x.get("Code") or x.get("code") or "").strip().upper()
        typ=str(x.get("Type") or x.get("type") or "").lower()
        if not code:
            continue
        if typ and not any(w in typ for w in ["common", "stock", "equity"]):
            continue
        ticker=code if "." in code else f"{code}.US"
        rows.append({
            "ticker": ticker,
            "symbol": re.sub(r"[^A-Z0-9_]+","_",ticker.replace(".","_")),
            "delisted": bool(delisted),
            "name": x.get("Name") or x.get("name") or "",
            "type": x.get("Type") or x.get("type") or "",
        })
    return rows

def fetch_eod(ticker):
    url=f"https://eodhd.com/api/eod/{ticker}"
    data=get_json(url, {"api_token":API, "fmt":"json", "period":"d", "from":START})
    if not data:
        return None
    df=pd.DataFrame(data)
    if df.empty or "date" not in df.columns:
        return None
    df["date"]=pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","adjusted_close","volume"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c], errors="coerce")
    df=df.dropna(subset=["date","open","high","low","close","volume"]).sort_values("date")
    if "adjusted_close" in df.columns:
        ratio=(df["adjusted_close"]/df["close"]).replace([float("inf"),float("-inf")], pd.NA)
        for c in ["open","high","low","close"]:
            df[c]=df[c]*ratio
        df["close"]=df["adjusted_close"]
    df=df[["date","open","high","low","close","volume"]].dropna()
    df=df[(df["open"]>0)&(df["high"]>0)&(df["low"]>0)&(df["close"]>0)&(df["volume"]>0)]
    return df.set_index("date")

print("fetching symbol lists", flush=True)
active=ticker_list(False)
dead=ticker_list(True)
allrows={r["ticker"]:r for r in active+dead}
rows=list(allrows.values())
if MAX_SYMBOLS:
    rows=rows[:MAX_SYMBOLS]

done=[]; failed=[]
for i,r in enumerate(rows,1):
    path=OUT/f"{r['symbol']}.parquet"
    try:
        df=fetch_eod(r["ticker"])
        if df is None or len(df)<20:
            failed.append({**r,"error":"too_few_rows"})
        else:
            df.to_parquet(path)
            done.append(r)
    except Exception as e:
        failed.append({**r,"error":repr(e)[:200]})
    print("progress",i,"done",len(done),"failed",len(failed),r["ticker"],flush=True)
    time.sleep(SLEEP)

db=Path("data/pit_database.db")
con=sqlite3.connect(db)
con.execute("""
CREATE TABLE IF NOT EXISTS universe_membership (
 symbol TEXT NOT NULL, index_name TEXT NOT NULL, added_date DATE NOT NULL,
 removed_date DATE, removal_reason TEXT, PRIMARY KEY(symbol,index_name,added_date)
)
""")
con.execute("DELETE FROM universe_membership WHERE index_name=?", (INDEX_NAME,))
for r in done:
    p=OUT/f"{r['symbol']}.parquet"
    if not p.exists():
        continue
    df=pd.read_parquet(p)
    first=str(pd.to_datetime(df.index).min().date())
    last=str(pd.to_datetime(df.index).max().date())
    removed=last if r["delisted"] else None
    con.execute("INSERT OR REPLACE INTO universe_membership VALUES (?,?,?,?,?)",
                (r["symbol"], INDEX_NAME, first, removed, "DELISTED" if removed else None))
con.commit()
con.close()

summary={
    "exchange":EXCHANGE,
    "active_listed":len(active),
    "delisted_listed":len(dead),
    "attempted":len(rows),
    "downloaded":len(done),
    "failed":len(failed),
    "failed_sample":failed[:5],
    "out":str(OUT),
    "membership_index":INDEX_NAME,
    "note":"Free EODHD plan is 20 calls/day and past-year only; this validates pipeline but cannot build full 2000-2026 survivorship-clean data."
}
REPORT.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
