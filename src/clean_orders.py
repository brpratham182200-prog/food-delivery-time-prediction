"""Clean the Kaggle food-delivery orders and join them with merged weather.

Outputs
  data/processed/orders_clean.csv  every cleaned order from WEATHER_START on
  data/processed/model_data.csv    subset in cities that have weather data
                                   (Bangalore, Mumbai, Pune, Chennai, Kolkata)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "food_delivery" / "train.csv"
WEATHER = ROOT / "data" / "processed" / "weather_merged.csv"
OUT_ALL = ROOT / "data" / "processed" / "orders_clean.csv"
OUT_MODEL = ROOT / "data" / "processed" / "model_data.csv"

# first day covered by the weather data; earlier orders are dropped
WEATHER_START = pd.Timestamp("2022-02-19")

# Delivery_person_ID prefix (before "RES") -> city
CITY_CODES = {
    "JAP": "Jaipur", "RANCHI": "Ranchi", "BANG": "Bangalore", "SUR": "Surat",
    "HYD": "Hyderabad", "MUM": "Mumbai", "MYS": "Mysore", "COIMB": "Coimbatore",
    "VAD": "Vadodara", "INDO": "Indore", "CHEN": "Chennai", "PUNE": "Pune",
    "AGR": "Agra", "LUDH": "Ludhiana", "KNP": "Kanpur", "ALH": "Allahabad",
    "DEH": "Dehradun", "GOA": "Goa", "AURG": "Aurangabad", "KOC": "Kochi",
    "KOL": "Kolkata", "BHP": "Bhopal",
}

# national / large regional festivals inside the data window
HOLIDAYS = {
    pd.Timestamp("2022-03-01"): "Maha Shivaratri",
    pd.Timestamp("2022-03-18"): "Holi",
    pd.Timestamp("2022-04-02"): "Ugadi / Gudi Padwa",
}

PEAK_HOURS = set(range(12, 15)) | set(range(19, 23))  # lunch, dinner


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. whitespace + textual NaN
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip().replace({"NaN": pd.NA, "": pd.NA})

    # 2. prefixes in values
    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    df["Weatherconditions"] = df["Weatherconditions"].replace({"NaN": pd.NA})
    df["Time_taken(min)"] = df["Time_taken(min)"].str.extract(r"(\d+)")[0].astype(int)

    # 3. typos / dtypes
    df["City"] = df["City"].replace({"Metropolitian": "Metropolitan"})
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    # ratings are on a 1-5 scale; 41 rows have 6.0 (all with age 50 -> placeholder rows)
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan

    # 4. city name from rider ID
    code = df["Delivery_person_ID"].str.extract(r"^(\w+?)RES")[0]
    df["city_name"] = code.map(CITY_CODES)

    # 5. coordinates: negative signs are data-entry errors; ~0 means missing
    coord_cols = ["Restaurant_latitude", "Restaurant_longitude",
                  "Delivery_location_latitude", "Delivery_location_longitude"]
    df[coord_cols] = df[coord_cols].abs()
    bad = (df["Restaurant_latitude"] < 1) | (df["Restaurant_longitude"] < 1)
    df.loc[bad, coord_cols] = np.nan
    df["distance_km"] = haversine_km(df.Restaurant_latitude, df.Restaurant_longitude,
                                     df.Delivery_location_latitude, df.Delivery_location_longitude)

    # 6. times: order time missing -> fall back to pickup time; handle midnight wrap
    ordered = pd.to_datetime(df["Order_Date"].dt.strftime("%Y-%m-%d") + " " + df["Time_Orderd"],
                             errors="coerce")
    picked = pd.to_datetime(df["Order_Date"].dt.strftime("%Y-%m-%d") + " " + df["Time_Order_picked"],
                            errors="coerce")
    picked = picked.where(ordered.isna() | (picked >= ordered), picked + pd.Timedelta(days=1))
    df["prep_time_min"] = (picked - ordered).dt.total_seconds() / 60
    order_ts = ordered.fillna(picked)
    df["order_hour"] = order_ts.dt.hour
    df["order_ts"] = order_ts

    # 7. calendar features
    df["day_of_week"] = df["Order_Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak"] = df["order_hour"].isin(PEAK_HOURS).astype(int)
    df["holiday_name"] = df["Order_Date"].map(HOLIDAYS)
    df["is_holiday"] = df["holiday_name"].notna().astype(int)

    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median for numeric, mode for categorical (fit on the data passed in)."""
    df = df.copy()
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries",
              "prep_time_min", "distance_km"]:
        df[c] = df[c].fillna(df[c].median())
    for c in ["Weatherconditions", "Road_traffic_density", "Festival", "City"]:
        df[c] = df[c].fillna(df[c].mode()[0])
    return df


def join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    # wind gusts are absent for Feb-Jul 2022 (the whole order window) -> drop
    w = weather.drop(columns=["latitude", "longitude", "datetime_ist", "source",
                              "wind_gust_max_kmh"])
    w["date"] = pd.to_datetime(w["date"])
    return df.merge(w, how="left",
                    left_on=["city_name", "Order_Date", "order_hour"],
                    right_on=["city", "date", "hour"]).drop(columns=["city", "date", "hour"])


def main():
    raw = pd.read_csv(RAW)
    df = clean_orders(raw)
    n0 = len(df)
    df = df[df["Order_Date"] >= WEATHER_START]
    print(f"raw {n0:,} -> {len(df):,} orders from {WEATHER_START.date()} on")

    df = join_weather(df, pd.read_csv(WEATHER))
    df["has_weather"] = df["temp_c"].notna().astype(int)
    df = impute(df)

    df.to_csv(OUT_ALL, index=False)
    model = df[df["has_weather"] == 1]
    model.to_csv(OUT_MODEL, index=False)
    print(f"orders_clean.csv {len(df):,} rows | model_data.csv {len(model):,} rows")


if __name__ == "__main__":
    main()
