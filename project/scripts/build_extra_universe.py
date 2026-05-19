"""Append symbols from data/features_26yr_liquid/ that pass ADV >= $3.5M
into data/universe_extra.txt (preserves existing non-symbol lines)."""

from pathlib import Path
import pandas as pd

FEATURES_DIR = Path("data/features_26yr_liquid")
EXTRA_FILE   = Path("data/universe_extra.txt")
MIN_ADV_USD  = 3_500_000.0
EXCLUDE_TOKENS = (".NS", ".BO", ".NSE", ".BSE")  # NSE/BSE handled by separate file
ADV_WINDOW   = 20

def norm_sym(p: Path) -> str:
    return p.stem.replace("_US", "").replace(".US", "").upper()

def adv_for(p: Path) -> float:
    try:
        df = pd.read_parquet(p, columns=["close", "volume"])
    except Exception:
        try:
            df = pd.read_parquet(p)
        except Exception:
            return 0.0
        if not {"close", "volume"}.issubset(df.columns):
            return 0.0
        df = df[["close", "volume"]]
    df = df.dropna()
    if df.empty:
        return 0.0
    dv = (df["close"].astype(float) * df["volume"].astype(float)).tail(ADV_WINDOW)
    return float(dv.mean()) if len(dv) else 0.0

def main():
    if not FEATURES_DIR.exists():
        raise SystemExit(f"missing {FEATURES_DIR}")
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    kept, dropped_liq, dropped_excl = [], [], []
    for p in files:
        sym = norm_sym(p)
        if any(tok in p.name for tok in EXCLUDE_TOKENS):
            dropped_excl.append(sym)
            continue
        adv = adv_for(p)
        if adv >= MIN_ADV_USD:
            kept.append((sym, adv))
        else:
            dropped_liq.append((sym, adv))

    kept_syms = sorted({s for s, _ in kept})

    # Preserve existing comment / non-symbol lines, drop existing symbol lines
    existing_meta = []
    if EXTRA_FILE.exists():
        for line in EXTRA_FILE.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                existing_meta.append(line)

    out_lines = []
    if existing_meta:
        out_lines.extend(existing_meta)
    else:
        out_lines.append("# Auto-managed: symbols from data/features_26yr_liquid passing ADV >= $3.5M")
    out_lines.append(f"# Generated: {len(kept_syms)} symbols (min_adv_usd={int(MIN_ADV_USD)})")
    out_lines.extend(kept_syms)
    EXTRA_FILE.write_text("\n".join(out_lines) + "\n")

    print(f"scanned       : {len(files)}")
    print(f"excluded(NS/BO): {len(dropped_excl)}")
    print(f"below_adv     : {len(dropped_liq)}")
    print(f"kept          : {len(kept_syms)}")
    print(f"wrote         : {EXTRA_FILE}")

if __name__ == "__main__":
    main()
