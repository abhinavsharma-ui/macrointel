import json, sqlite3, os
from pathlib import Path
import pandas as pd

ROOT=Path(os.getenv("SURV_FEATURE_ROOT","data/features_26yr"))
OUT=Path(os.getenv("SURV_OUT","reports/ohlcv_survivorship_membership.json"))
DB=Path("data/pit_database.db")
INDEX_NAME=os.getenv("SURV_INDEX_NAME","FEATURE_26YR_AVAILABILITY")
REMOVE_BUFFER_DAYS=int(os.getenv("SURV_REMOVE_BUFFER_DAYS","45"))

def parse_date(raw):
    for c in ["date","timestamp","datetime","time","Date"]:
        if c in raw.columns:
            s=pd.Series(pd.to_datetime(raw[c], errors="coerce", utc=True))
            if s.notna().sum() and s.dt.year.median()>1980:
                return s.dt.tz_localize(None).dt.normalize()
    s=pd.Series(pd.to_datetime(raw.index, errors="coerce", utc=True))
    if s.notna().sum() and s.dt.year.median()>1980:
        return s.dt.tz_localize(None).dt.normalize()
    raise ValueError("no usable date/index")

rows=[]
for i,p in enumerate(sorted(ROOT.glob("*.parquet")),1):
    try:
        df=pd.read_parquet(p, columns=None)
        d=parse_date(df).dropna()
        if len(d):
            rows.append({"symbol":p.stem.upper(),"first":str(d.min().date()),"last":str(d.max().date()),"rows":int(len(d))})
    except Exception as e:
        if i < 20:
            print("skip",p.name,repr(e),flush=True)
    if i%250==0:
        print("scanned",i,"ok",len(rows),flush=True)

if not rows:
    raise SystemExit(f"no usable symbols under {ROOT}")

cov=pd.DataFrame(rows)
cov["first_dt"]=pd.to_datetime(cov["first"])
cov["last_dt"]=pd.to_datetime(cov["last"])
global_last=cov["last_dt"].max()
survives=cov["last_dt"] >= global_last-pd.Timedelta(days=REMOVE_BUFFER_DAYS)
cov["removed_date"]=None
cov.loc[~survives,"removed_date"]=cov.loc[~survives,"last"]

DB.parent.mkdir(parents=True,exist_ok=True)
con=sqlite3.connect(DB)
con.execute("""
CREATE TABLE IF NOT EXISTS universe_membership (
    symbol TEXT NOT NULL,
    index_name TEXT NOT NULL,
    added_date DATE NOT NULL,
    removed_date DATE,
    removal_reason TEXT,
    PRIMARY KEY (symbol, index_name, added_date)
)
""")
con.execute("DELETE FROM universe_membership WHERE index_name=?", (INDEX_NAME,))
for _,r in cov.iterrows():
    con.execute(
        "INSERT OR REPLACE INTO universe_membership(symbol,index_name,added_date,removed_date,removal_reason) VALUES (?,?,?,?,?)",
        (r.symbol, INDEX_NAME, r.first, r.removed_date, "DATA_ENDED" if r.removed_date else None),
    )
con.commit()
con.close()

summary={
    "status":"PASS_PROXY_HAS_NON_SURVIVORS" if int((~survives).sum()) else "FAIL_NO_NON_SURVIVORS_IN_ROOT",
    "root":str(ROOT),
    "index_name":INDEX_NAME,
    "symbols":int(len(cov)),
    "global_first":str(cov["first_dt"].min().date()),
    "global_last":str(global_last.date()),
    "survivor_symbols":int(survives.sum()),
    "non_survivor_symbols":int((~survives).sum()),
    "late_starters_after_2015":int((cov["first_dt"]>=pd.Timestamp("2015-01-01")).sum()),
    "pit_database":str(DB),
    "membership_rows_written":int(len(cov)),
}
OUT.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
