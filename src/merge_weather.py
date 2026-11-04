"""Merge the two Open-Meteo exports into one tidy weather table.

Inputs (Open-Meteo multi-location CSV: 3 sections separated by blank lines
-> locations, hourly, daily):
  * "2022 feb-july.csv"                  19-Feb-2022 .. 02-Jul-2022, times in GMT
  * "open-meteo-12.97N77.56E918m.csv"    01-Jul-2022 .. 2026,       times in IST

Output: data/processed/weather_merged.csv
  one row per (city, IST hour), with that day's daily aggregates attached.

Gotchas handled:
  * location_id order differs between files -> cities mapped by coordinates
  * GMT -> IST conversion (+5:30, floored to the hour to match the IST export)
  * overlapping days (1-2 Jul 2022) -> IST export wins
  * wind_gusts missing in the Feb-Jul export -> NaN
"""
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "weather"
OUT = ROOT / "data" / "processed" / "weather_merged.csv"

CITY_COORDS = {  # (lat, lon) as exported by Open-Meteo
    "Bangalore": (12.970123, 77.56364),
    "Mumbai": (19.086115, 72.85291),
    "Delhi": (28.646748, 77.2748),
    "Pune": (18.523726, 73.86876),
    "Chennai": (13.110721, 80.2459),
    "Kolkata": (22.530754, 88.3195),
}

RENAME = {
    "temperature_2m (°C)": "temp_c",
    "precipitation_sum (mm)": "precip_mm_day",
    "rain_sum (mm)": "rain_mm_day",
    "temperature_2m_max (°C)": "temp_max_c",
    "temperature_2m_min (°C)": "temp_min_c",
    "wind_speed_10m_max (km/h)": "wind_max_kmh",
    "wind_gusts_10m_max (km/h)": "wind_gust_max_kmh",
}


def read_sections(path: Path):
    text = path.read_text(encoding="utf-8")
    blocks = [b for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    locs, hourly, daily = (pd.read_csv(StringIO(b)) for b in blocks)
    return locs, hourly, daily


def city_lookup(locs: pd.DataFrame) -> dict:
    out = {}
    for _, r in locs.iterrows():
        city = min(
            CITY_COORDS,
            key=lambda c: (CITY_COORDS[c][0] - r.latitude) ** 2
            + (CITY_COORDS[c][1] - r.longitude) ** 2,
        )
        out[r.location_id] = city
    return out


def load(path: Path, source: str):
    locs, hourly, daily = read_sections(path)
    cities = city_lookup(locs)
    offset = int(locs.utc_offset_seconds.iloc[0])

    hourly = hourly.rename(columns=RENAME)
    hourly["city"] = hourly.location_id.map(cities)
    t = pd.to_datetime(hourly.time)
    if offset == 0:  # GMT export -> IST
        t = (t + pd.Timedelta(hours=5, minutes=30)).dt.floor("h")
    hourly["datetime_ist"] = t
    hourly = hourly[["city", "datetime_ist", "temp_c"]].assign(source=source)

    daily = daily.rename(columns=RENAME)
    daily["city"] = daily.location_id.map(cities)
    daily["date"] = pd.to_datetime(daily.time).dt.date
    daily = daily.drop(columns=["location_id", "time"]).assign(source=source)
    return hourly, daily


def main():
    h_new, d_new = load(RAW / "2022 feb-july.csv", "feb_jul_2022")
    h_old, d_old = load(RAW / "open-meteo-12.97N77.56E918m.csv", "jul2022_onward")

    # IST export listed last so keep="last" prefers it on overlapping keys
    hourly = (
        pd.concat([h_new, h_old])
        .drop_duplicates(["city", "datetime_ist"], keep="last")
    )
    daily = (
        pd.concat([d_new, d_old])
        .drop_duplicates(["city", "date"], keep="last")
        .drop(columns="source")
    )

    hourly["date"] = hourly.datetime_ist.dt.date
    hourly["hour"] = hourly.datetime_ist.dt.hour
    merged = hourly.merge(daily, on=["city", "date"], how="left")
    merged["latitude"] = merged.city.map(lambda c: CITY_COORDS[c][0])
    merged["longitude"] = merged.city.map(lambda c: CITY_COORDS[c][1])
    merged = merged.sort_values(["city", "datetime_ist"])[
        ["city", "latitude", "longitude", "datetime_ist", "date", "hour",
         "temp_c", "precip_mm_day", "rain_mm_day", "temp_max_c", "temp_min_c",
         "wind_max_kmh", "wind_gust_max_kmh", "source"]
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"Wrote {len(merged):,} rows -> {OUT}")
    print(merged.groupby("city").datetime_ist.agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
