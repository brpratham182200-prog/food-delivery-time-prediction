# Predicting Food Delivery Time in Indian Cities
### The impact of weather, traffic and festivals

Online food delivery platforms (Zomato, Swiggy) promise fast delivery, but actual
delivery times vary widely with weather, traffic, festival rush and rider factors.
This project builds regression models that predict **delivery time in minutes**.

**Objectives**
1. Clean and combine messy real-world data from multiple food-delivery and weather sources.
2. Analyse and visualise which factors affect delivery time most.
3. Build regression models that predict delivery time from order, rider, weather and traffic details.
4. Compare models (Linear, Ridge, Lasso, Random Forest) using R², MAE and RMSE.

---

## Architecture

```
 RAW SOURCES                 CLEANING                    FEATURES                     MODELING                    OUTPUT
 ───────────                 ────────                    ────────                     ────────                    ──────
 food_delivery/train.csv ──► clean_orders() ──┐
   (45,593 orders,                            │
    Feb–Apr 2022)                             ├─► join on (city, date, hour) ─► build_features() ─► 80/20 split
                                              │                                      │                  │
 weather/*.csv (Open-Meteo) ► merge_weather() ┘                                      │                  ├─► Linear Regression
   2 exports, GMT + IST       → weather_merged.csv                                   │                  ├─► Ridge  (GridSearch α)
                                                                                     │                  ├─► Lasso  (GridSearch α)
 Indian holiday calendar ───────────────────────────────► festival flags ────────────┘                  └─► Random Forest (GridSearch)
                                                                                                                │
 zomato/zomato.csv ────────► EDA only (Bangalore restaurant cost / rating context)              5-fold CV · R² · MAE · RMSE
                                                                                                                │
                                                                                     comparison table · residual plots · feature importance
```

## Repository layout

```
├── data/
│   ├── raw/
│   │   ├── food_delivery/        train.csv, test.csv, Sample_Submission.csv
│   │   ├── weather/              Open-Meteo exports (Feb–Jul 2022 GMT; Jul 2022–2026 IST)
│   │   └── zomato/               zomato.csv  (not committed – 548 MB, see below)
│   └── processed/
│       └── weather_merged.csv    output of src/merge_weather.py
├── notebooks/
│   ├── 01_cleaning_merging.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── src/
│   └── merge_weather.py          merges both weather exports into one tidy table
├── reports/figures/
└── requirements.txt
```

## Data sources

| Source | Content | Use |
|---|---|---|
| [Food Delivery Dataset (Kaggle)](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset) | Rider age/rating, restaurant & customer lat/long, order/pickup time, weather, traffic, festival, city, vehicle, `Time_taken(min)` | Main dataset & target |
| [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api) | Hourly temperature; daily precipitation, rain, min/max temp, max wind for Bangalore, Mumbai, Delhi, Pune, Chennai, Kolkata | Numeric weather features |
| [Zomato Bangalore Restaurants (Kaggle)](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) | Restaurant cost, rating, cuisine, area | EDA context only |

`zomato.csv` is too large for GitHub – download it from Kaggle into `data/raw/zomato/`.

### Weather merge (`src/merge_weather.py`)
The two Open-Meteo exports differ in ways that silently break a naive concat:
- **Timezone** – Feb–Jul file is GMT, the other is IST → shifted +5:30.
- **location_id order** – Chennai/Kolkata IDs are swapped between files → cities matched by coordinates.
- **Overlap** – 1–2 Jul 2022 appears in both → IST export kept.
- **Missing column** – `wind_gusts_10m_max` absent in the Feb–Jul export → NaN.

Result: 229,650 rows, one per city-hour, continuous from 19 Feb 2022 to 2 Jul 2026.

## Plan

**1 · Cleaning** – strip trailing spaces; `"NaN "` → NaN; remove `"conditions "` prefix in weather and
`"(min) "` prefix in target; fix `"Metropolitian"`; parse `DD-MM-YYYY` dates; fix zero/negative coordinates;
derive **city** from `Delivery_person_ID` prefix (`BANGRES…` → Bangalore); impute (median numeric, mode categorical).

**2 · Features** – Haversine `distance_km`; `prep_time_min` (pickup − order); `order_hour`, `day_of_week`,
`is_weekend`, `is_peak`; weather `temp_c`, `rain_mm_day`, `wind_max_kmh`; ordinal traffic (Low<Medium<High<Jam);
one-hot weather / vehicle / order type / city type; StandardScaler.

**3 · EDA** – target distribution; time by traffic, weather, festival, city type, vehicle; distance vs time;
correlation heatmap; time by hour of day; rider rating vs time; Zomato cost/rating by Bangalore area.

**4 · Modeling** – 80/20 split of `train.csv` (Kaggle `test.csv` has no labels); sklearn `Pipeline`
(preprocessor → model) to avoid leakage; Linear, Ridge, Lasso, Random Forest with `GridSearchCV`.

**5 · Evaluation** – R², MAE, RMSE on hold-out + 5-fold CV mean ± std; actual-vs-predicted and residual plots;
RF feature importance vs Lasso coefficients.

## Known limitations
- Weather covers 6 cities (5 overlap with orders, ≈29 % of rows) and starts 19 Feb 2022, so orders from
  11–18 Feb have no weather match.
- Daily aggregates in the Feb–Jul export are computed on GMT days.
- Delivery coordinates in the Kaggle dataset are partially synthetic.

## Setup
```bash
pip install -r requirements.txt
python src/merge_weather.py
```
