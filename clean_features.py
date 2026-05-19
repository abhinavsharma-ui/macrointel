"""
clean_features.py — strip dead/harmful/redundant columns from the parquet store.
Run once. Idempotent — safe to re-run.
"""
import glob, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "project"))
from pipeline.new_features import DEAD_FEATURES, HARMFUL_FEATURES, REDUNDANT_FEATURES

TO_DROP = set(DEAD_FEATURES + HARMFUL_FEATURES + REDUNDANT_FEATURES)
for dir_name in ("project/data/features", "project/data/features_10yr"):
    pattern = os.path.join(HERE, dir_name, "*.parquet")
    files = glob.glob(pattern)
    if not files:
        continue
    print(f"\n{dir_name}: {len(files)} files")
    total_dropped = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
            drop_here = [c for c in TO_DROP if c in df.columns]
            if drop_here:
                df.drop(columns=drop_here).to_parquet(f, index=True)
                total_dropped += len(drop_here)
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}")
    print(f"  removed {total_dropped} total column-instances across {len(files)} files")
print("\nDone.")
