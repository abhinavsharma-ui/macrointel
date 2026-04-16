import re

filepath = "project/build_finnhub_features.py"
with open(filepath, "r") as f:
    content = f.read()

# Fix 1: normalize_daily_index - always strip timezone unconditionally
old_fn = '''def _normalize_daily_index(values) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()'''

new_fn = '''def _normalize_daily_index(values) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    idx = idx.tz_localize(None)
    idx = idx.normalize()
    try:
        idx = idx.as_unit("ns")
    except AttributeError:
        idx = pd.DatetimeIndex(idx.astype("datetime64[ns]"))
    return idx'''

if old_fn in content:
    content = content.replace(old_fn, new_fn)
    print("✓ Fixed _normalize_daily_index")
else:
    print("⚠ _normalize_daily_index pattern not found (may already be patched)")

# Fix 2: deprecated reindex method=ffill
old_reindex = '.reindex(df.index, method="ffill")'
new_reindex = '.reindex(df.index).ffill()'
count = content.count(old_reindex)
content = content.replace(old_reindex, new_reindex)
print(f"✓ Fixed {count} reindex calls")

with open(filepath, "w") as f:
    f.write(content)

print("Done — re-run: python project/build_finnhub_features.py --workers 1 --symbols AAPL MSFT TSLA")
