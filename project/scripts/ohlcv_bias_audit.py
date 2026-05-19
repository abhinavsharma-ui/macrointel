import json, os, re, sqlite3, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATURE_ROOT=Path(os.getenv("BIAS_FEATURE_ROOT","data/features_26yr_liquid"))
OUT=Path(os.getenv("BIAS_AUDIT_OUT","reports/ohlcv_bias_audit.json"))
HOLD=int(os.getenv("FIXED_HOLD_DAYS","10"))
COST=float(os.getenv("FIXED_TOTAL_COST_PCT","0.45"))
MIN_ADV=float(os.getenv("FIXED_MIN_ADV20_DOLLAR_VOL","5000000"))
MIN_ENTRY_PRICE=float(os.getenv("FIXED_MIN_ENTRY_PRICE","5"))
MAX_ABS_RET=float(os.getenv("FIXED_MAX_ABS_HONEST_RET_PCT","100"))
MAX_DAILY_JUMP=float(os.getenv("FIXED_MAX_DAILY_JUMP_PCT","60"))
MAX_ENTRY_GAP=float(os.getenv("FIXED_MAX_ENTRY_GAP_PCT","40"))
SEED=42

BAD_FEATURE_TOKENS=("gross_ret","future","next","target","label","edge","drawdown","exit","hold_days")
EVENT_TOKENS=("event","earnings","filing","official","sentiment","news","peer","alpha_signal")

def log(*x): print("BIAS_AUDIT", *x, flush=True)

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, tuple): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (pd.Timestamp,)): return str(x.date())
    if pd.isna(x) if not isinstance(x, (dict,list,tuple,str)) else False: return None
    return x

def parse_date(raw):
    for c in ["date","timestamp","datetime","time","Date"]:
        if c in raw.columns:
            s=pd.Series(pd.to_datetime(raw[c],errors="coerce",utc=True))
            if s.notna().sum()>0 and s.dt.year.median()>1980:
                return s.dt.tz_localize(None).dt.normalize()
    s=pd.Series(pd.to_datetime(raw.index,errors="coerce",utc=True))
    if s.notna().sum()>0 and s.dt.year.median()>1980:
        return s.dt.tz_localize(None).dt.normalize()
    raise ValueError("no usable date/index")

def feature_cols(df):
    banned={"date","symbol","year","open","high","low","close","volume","entry_open","exit_close","exit_date","honest_ret","adv20_dollar_vol","entry_gap_pct","max_path_jump_pct"}
    out=[]
    for c in df.columns:
        low=str(c).lower()
        if c in banned or any(tok in low for tok in BAD_FEATURE_TOKENS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    preferred=["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]
    return [c for c in preferred if c in out]+[c for c in out if c not in preferred]

def load_panel():
    files=sorted(FEATURE_ROOT.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {FEATURE_ROOT}")
    parts=[]; coverage=[]; raw_suspicious_cols=set()
    for i,p in enumerate(files,1):
        try:
            raw=pd.read_parquet(p)
            raw_suspicious_cols.update([c for c in raw.columns if any(tok in str(c).lower() for tok in BAD_FEATURE_TOKENS)])
            if not {"open","close","volume"}.issubset(raw.columns):
                continue
            x=raw.copy()
            x["date"]=parse_date(raw).to_numpy()
            x["symbol"]=p.stem
            for c in ["open","high","low","close","volume"]:
                if c in x.columns:
                    x[c]=pd.to_numeric(x[c],errors="coerce")
            x=x.dropna(subset=["date","open","close","volume"]).sort_values("date").reset_index(drop=True)
            x=x[(x["open"]>0)&(x["close"]>0)&(x["volume"]>0)]
            if x.empty:
                continue
            coverage.append({"symbol":p.stem,"first":str(pd.Timestamp(x.date.min()).date()),"last":str(pd.Timestamp(x.date.max()).date()),"rows":len(x)})
            x["adv20_dollar_vol"]=(x["close"]*x["volume"]).rolling(20,min_periods=5).mean()
            parts.append(x)
        except Exception as e:
            log("load_error",p.name,repr(e))
        if i%100==0 or i==len(files):
            log("loaded",i,"ok",len(parts))
    df=pd.concat(parts,ignore_index=True).sort_values(["symbol","date"]).reset_index(drop=True)
    before=len(df)
    df=df[df["adv20_dollar_vol"].fillna(0)>=MIN_ADV].copy()

    g=df.groupby("symbol",sort=False)
    df["entry_open"]=g["open"].shift(-1)
    df["exit_close"]=g["close"].shift(-(HOLD+1))
    df["exit_date"]=g["date"].shift(-(HOLD+1))
    df["honest_ret"]=((df["exit_close"]/df["entry_open"])-1.0)*100.0
    df["entry_gap_pct"]=((df["entry_open"]/df["close"])-1.0).abs()*100.0
    daily_jump=g["close"].pct_change().abs()*100.0
    df["max_path_jump_pct"]=daily_jump.groupby(df["symbol"],sort=False).transform(lambda s: s.shift(-1).rolling(HOLD+1,min_periods=1).max().shift(-HOLD))

    before_sanity=len(df)
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=["entry_open","exit_close","exit_date","honest_ret","entry_gap_pct","max_path_jump_pct"])
    df=df[(df["entry_open"]>=MIN_ENTRY_PRICE)&(df["exit_close"]>=MIN_ENTRY_PRICE)]
    df=df[df["honest_ret"].abs()<=MAX_ABS_RET]
    df=df[df["entry_gap_pct"]<=MAX_ENTRY_GAP]
    df=df[df["max_path_jump_pct"]<=MAX_DAILY_JUMP]
    log("rows_after_liquidity",len(df),"raw_suspicious_cols",len(raw_suspicious_cols))
    return df.sort_values(["date","symbol"]).reset_index(drop=True), coverage, sorted(raw_suspicious_cols)

def scan_sources():
    patterns={
        "negative_shift": re.compile(r"\.shift\s*\(\s*-\d+"),
        "centered_rolling": re.compile(r"rolling\s*\([^)]*center\s*=\s*True"),
        "backfill": re.compile(r"\.bfill\(|method\s*=\s*['\"]bfill['\"]|backfill"),
        "future_token": re.compile(r"future|lookahead|look_ahead", re.I),
        "target_label_token": re.compile(r"target|label|gross_ret|edge_pct|drawdown_pct|exit_timestamp", re.I),
    }
    scan_paths=[]
    for root in ["pipeline","scripts","models"]:
        scan_paths += list(Path(root).glob("**/*.py"))
    scan_paths += list(Path(".").glob("*.py"))
    hits=[]
    for path in sorted(set(scan_paths)):
        try:
            lines=path.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        for n,line in enumerate(lines,1):
            for name,pat in patterns.items():
                if pat.search(line):
                    hits.append({"file":str(path),"line":n,"pattern":name,"text":line.strip()[:240]})
    feature_builder_hits=[h for h in hits if any(x in h["file"] for x in ["feature_engineering.py","build_feature","event_alpha","earnings","sentiment"])]
    return hits[:400], feature_builder_hits[:200]

def model_leak_smoke(df, fs):
    from sklearn.utils import shuffle
    dates=np.array(sorted(df.date.drop_duplicates()))
    cut=dates[int(len(dates)*0.70)]
    train=df[df.date<cut].copy()
    test=df[df.date>=cut].copy()
    if len(train)>200000:
        train=train.sample(200000,random_state=SEED)
    if len(test)>150000:
        test=test.sample(150000,random_state=SEED+1)
    y=(train.honest_ret>COST).astype(int)
    yt=(test.honest_ret>COST).astype(int)
    X=train[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    Xt=test[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    model=HistGradientBoostingClassifier(max_iter=80,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.1,random_state=SEED)
    model.fit(X,y)
    auc=float(roc_auc_score(yt,model.predict_proba(Xt)[:,1]))

    ys=shuffle(y,random_state=SEED+99).reset_index(drop=True)
    model2=HistGradientBoostingClassifier(max_iter=80,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.1,random_state=SEED+2)
    model2.fit(X.reset_index(drop=True),ys)
    shuffled_auc=float(roc_auc_score(yt,model2.predict_proba(Xt)[:,1]))

    sample=df.sample(min(len(df),250000),random_state=SEED)
    corrs={}
    target=sample["honest_ret"]
    for c in fs:
        s=pd.to_numeric(sample[c],errors="coerce")
        if s.notna().sum()>1000 and s.nunique(dropna=True)>5:
            val=s.corr(target,method="spearman")
            if pd.notna(val):
                corrs[c]=float(val)
    top_corr=sorted(corrs.items(),key=lambda kv:abs(kv[1]),reverse=True)[:25]
    return {"auc":auc,"shuffled_label_auc":shuffled_auc,"top_abs_spearman_to_honest_ret":top_corr}

def survivorship_audit(coverage):
    cov=pd.DataFrame(coverage)
    cov["first"]=pd.to_datetime(cov["first"])
    cov["last"]=pd.to_datetime(cov["last"])
    global_first=str(cov["first"].min().date())
    global_last=pd.Timestamp(cov["last"].max())
    survive_to_end=(cov["last"] >= global_last-pd.Timedelta(days=45))
    ended_before_end=int((~survive_to_end).sum())
    started_after_2015=int((cov["first"] >= pd.Timestamp("2015-01-01")).sum())

    pit={"exists":Path("data/pit_database.db").exists(),"universe_membership_rows":0,"status":"not_checked"}
    if pit["exists"]:
        try:
            con=sqlite3.connect("data/pit_database.db")
            pit["universe_membership_rows"]=int(con.execute("select count(*) from universe_membership").fetchone()[0])
            pit["status"]="present_with_rows" if pit["universe_membership_rows"] else "present_empty"
            con.close()
        except Exception as e:
            pit["status"]=f"error:{e}"

    status="FAIL_NOT_POINT_IN_TIME"
    reason="optimizer used a fixed parquet file universe; no historical membership filter was applied"
    if pit["universe_membership_rows"]>0:
        status="WARN_PIT_DB_EXISTS_BUT_OPTIMIZER_NOT_USING_IT"
        reason="PIT universe table exists, but this audit/optimizer did not filter symbols by membership date"

    return {
        "status":status,
        "reason":reason,
        "feature_root":str(FEATURE_ROOT),
        "symbols":int(len(cov)),
        "global_first":global_first,
        "global_last":str(global_last.date()),
        "symbols_ending_before_global_last_minus_45d":ended_before_end,
        "symbols_starting_after_2015":started_after_2015,
        "pit_database":pit,
        "interpretation":"survivorship bias is not eliminated unless entries are limited to symbols that were tradable/members as of each signal date, including delisted names"
    }

def main():
    df,coverage,raw_suspicious_cols=load_panel()
    fs=feature_cols(df)
    trained_suspicious=[c for c in fs if any(tok in str(c).lower() for tok in BAD_FEATURE_TOKENS)]
    event_like=[c for c in fs if any(tok in str(c).lower() for tok in EVENT_TOKENS)]
    event_nonzero={}
    for c in event_like:
        s=pd.to_numeric(df[c],errors="coerce").fillna(0)
        event_nonzero[c]=round(float((s.abs()>1e-12).mean()*100),4)

    source_hits, feature_builder_hits=scan_sources()
    smoke=model_leak_smoke(df,fs)
    surv=survivorship_audit(coverage)

    feature_status="PASS_NO_TARGET_NAMED_TRAIN_FEATURES" if not trained_suspicious else "FAIL_TARGET_NAMED_FEATURES_USED"
    lookahead_status="WARN_REVIEW_SOURCE_HITS" if feature_builder_hits else "PASS_NO_OBVIOUS_FEATURE_BUILDER_SHIFT_CENTER_BFILL_HITS"
    if smoke["shuffled_label_auc"]>0.56:
        lookahead_status="FAIL_SHUFFLED_LABEL_TEST_TOO_GOOD"

    out={
        "feature_root":str(FEATURE_ROOT),
        "rows":len(df),
        "symbols":int(df.symbol.nunique()),
        "trained_feature_count":len(fs),
        "feature_status":feature_status,
        "trained_suspicious_features":trained_suspicious,
        "raw_suspicious_columns_not_used":raw_suspicious_cols[:200],
        "lookahead_status":lookahead_status,
        "model_smoke":smoke,
        "event_like_feature_nonzero_pct":event_nonzero,
        "feature_builder_source_hits":feature_builder_hits,
        "source_hits_sample":source_hits,
        "survivorship":surv,
        "hard_truth":{
            "return_target":"cleaner: next open to HOLD+1 close, no gross_ret",
            "lookahead_bias":"not proven absent until feature_builder_source_hits are reviewed",
            "survivorship_bias":"not eliminated in this optimizer unless PIT universe membership is applied"
        }
    }
    OUT.write_text(json.dumps(clean(out),indent=2))
    print(json.dumps(clean({
        "feature_status":out["feature_status"],
        "lookahead_status":out["lookahead_status"],
        "auc":out["model_smoke"]["auc"],
        "shuffled_label_auc":out["model_smoke"]["shuffled_label_auc"],
        "trained_suspicious_features":out["trained_suspicious_features"],
        "feature_builder_hit_count":len(out["feature_builder_source_hits"]),
        "survivorship_status":out["survivorship"]["status"],
        "survivorship_reason":out["survivorship"]["reason"],
        "report":str(OUT)
    }),indent=2))

if __name__=="__main__":
    main()
