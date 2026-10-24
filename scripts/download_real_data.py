import csv
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_URL = "https://www.smard.de/app/chart_data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Series IDs documented by the public SMARD API.
# Day-ahead price is requested for the DE-LU bidding zone.
SERIES = {
    "price_eur_mwh": ("4169", "DE-LU"),
    "grid_load": ("410", "DE"),
    "residual_load": ("4359", "DE"),
    "wind_offshore": ("1225", "DE"),
    "hydro": ("1226", "DE"),
    "biomass": ("4066", "DE"),
    "wind_onshore": ("4067", "DE"),
    "solar": ("4068", "DE"),
    "hard_coal": ("4069", "DE"),
    "pumped_storage": ("4070", "DE"),
    "natural_gas": ("4071", "DE"),
    "other_conventional": ("1227", "DE"),
    "other_renewables": ("1228", "DE"),
}

# Download from this date through the latest available data.
START_DATE = "2020-01-01"
END_DATE = datetime.now(timezone.utc).date().isoformat()
RESOLUTION = "hour"
BERLIN = ZoneInfo("Europe/Berlin")


def get_json(url, attempts=4):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "energy-forecasting-research/1.0"}
    )
    last_error = None

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Request failed: {url}\n{last_error}")


def berlin_date(timestamp_ms):
    return datetime.fromtimestamp(
        timestamp_ms / 1000, tz=timezone.utc
    ).astimezone(BERLIN).date().isoformat()


def download_series(column, filter_id, region, start_date, end_date):
    index_url = (
        f"{BASE_URL}/{filter_id}/{region}/index_{RESOLUTION}.json"
    )
    index_data = get_json(index_url)
    timestamps = index_data.get("timestamps", [])

    if not timestamps:
        raise RuntimeError(
            f"No timestamps returned for {column}. URL: {index_url}"
        )

    result = {}
    all_ts = sorted(int(ts) for ts in timestamps)

    start_local = datetime.strptime(
        start_date, "%Y-%m-%d"
    ).replace(tzinfo=BERLIN)

    end_local = datetime.strptime(
        end_date, "%Y-%m-%d"
    ).replace(hour=23, minute=59, second=59, tzinfo=BERLIN)

    start_ms = int(
        start_local.astimezone(timezone.utc).timestamp() * 1000
    )
    end_ms = int(
        end_local.astimezone(timezone.utc).timestamp() * 1000
    )

    # Include the chunk immediately before the requested start date.
    previous_chunks = [ts for ts in all_ts if ts < start_ms]
    selected = (
        ([previous_chunks[-1]] if previous_chunks else [])
        + [ts for ts in all_ts if start_ms <= ts <= end_ms]
    )
    selected = sorted(set(selected))

    print(f"\n{column}: {len(selected)} data chunks available")

    for i, ts in enumerate(selected, start=1):
        url = (
            f"{BASE_URL}/{filter_id}/{region}/"
            f"{filter_id}_{region}_{RESOLUTION}_{ts}.json"
        )

        try:
            payload = get_json(url)
            for point in payload.get("series", []):
                if not point or len(point) < 2:
                    continue

                timestamp_ms, value = point[0], point[1]
                if timestamp_ms is None:
                    continue

                timestamp_ms = int(timestamp_ms)
                local_date = berlin_date(timestamp_ms)

                if start_date <= local_date <= end_date and value is not None:
                    result[timestamp_ms] = value

            if i % 25 == 0 or i == len(selected):
                print(f"  Downloaded chunks: {i}/{len(selected)}")

        except Exception as exc:
            print(f"  WARNING: skipped chunk {ts}: {exc}")

    if not result:
        print(f"  WARNING: no observations downloaded for {column}")

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SMARD REAL DATA DOWNLOADER")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Resolution: {RESOLUTION}")
    print(f"Output: {OUTPUT_DIR}")
    print("Source: SMARD / Bundesnetzagentur")

    all_series = {}
    for column, (filter_id, region) in SERIES.items():
        try:
            all_series[column] = download_series(
                column, filter_id, region, START_DATE, END_DATE
            )
        except Exception as exc:
            print(f"ERROR downloading {column}: {exc}")
            all_series[column] = {}

    timestamps = sorted(
        set().union(*(series.keys() for series in all_series.values()))
    )

    if not timestamps:
        raise RuntimeError(
            "No data downloaded. Check internet access and the SMARD API."
        )

    output_file = OUTPUT_DIR / "germany_electricity_hourly.csv"
    columns = ["timestamp_utc", "datetime_berlin"] + list(SERIES.keys())

    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for ts in timestamps:
            utc_dt = datetime.fromtimestamp(
                ts / 1000, tz=timezone.utc
            )
            berlin_dt = utc_dt.astimezone(BERLIN)

            row = {
                "timestamp_utc": utc_dt.isoformat(),
                "datetime_berlin": berlin_dt.isoformat(),
            }

            for name, values in all_series.items():
                row[name] = values.get(ts, "")

            writer.writerow(row)

    print("\nDOWNLOAD FINISHED")
    print(f"Output file: {output_file}")
    print(f"Rows: {len(timestamps):,}")
    print(f"Columns: {len(columns)}")

    print("\nObservations by series:")
    for name, values in all_series.items():
        print(f"  {name}: {len(values):,}")

    print(
        "\nIMPORTANT: inspect missing values, units, timestamps, "
        "and series coverage before model training."
    )


if __name__ == "__main__":
    main()
