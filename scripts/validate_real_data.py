import csv
from pathlib import Path
from datetime import datetime
from collections import Counter

path = Path("data/raw/germany_electricity_hourly.csv")

if not path.exists():
    raise FileNotFoundError(f"File not found: {path.resolve()}")

with path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("The CSV has no data rows.")

columns = list(rows[0].keys())

print("=" * 65)
print("REAL DATA QUALITY REPORT")
print("=" * 65)
print(f"File: {path.resolve()}")
print(f"Rows: {len(rows):,}")
print(f"Columns: {len(columns)}")
print("\nColumn names:")
for c in columns:
    print(" -", c)

print("\nDate range:")
for c in ("timestamp_utc", "datetime_berlin"):
    values = [r[c] for r in rows if r.get(c)]
    if values:
        print(f" {c}: {min(values)}  -->  {max(values)}")

print("\nMissing values by column:")
for c in columns:
    missing = sum(1 for r in rows if not (r.get(c) or "").strip())
    print(f" {c}: {missing:,} ({missing/len(rows):.3%})")

print("\nNumeric summary:")
for c in columns:
    if c in ("timestamp_utc", "datetime_berlin"):
        continue
    vals = []
    invalid = 0
    for r in rows:
        raw = (r.get(c) or "").strip()
        if not raw:
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            invalid += 1
    if vals:
        print(
            f" {c}: n={len(vals):,}, min={min(vals):.4f}, "
            f"max={max(vals):.4f}, mean={sum(vals)/len(vals):.4f}, "
            f"invalid={invalid}"
        )

timestamps = [r["timestamp_utc"] for r in rows if r.get("timestamp_utc")]
counts = Counter(timestamps)
duplicates = sum(n - 1 for n in counts.values() if n > 1)
print("\nTimestamp checks:")
print(f" Non-empty UTC timestamps: {len(timestamps):,}")
print(f" Duplicate UTC timestamps: {duplicates:,}")
print(f" Rows with empty UTC timestamp: {len(rows)-len(timestamps):,}")

print("\nFirst 3 rows:")
for r in rows[:3]:
    print(r)

print("\nLast 3 rows:")
for r in rows[-3:]:
    print(r)

print("\nReport complete. The original CSV was not modified.")
