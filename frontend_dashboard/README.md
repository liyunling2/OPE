# RestaurantIQ — Marketing Intelligence Dashboard

Streamlit dashboard that reads directly from your notebook pipeline outputs
and provides real-time restaurant performance analytics + AI-generated marketing strategies.

## Setup

```bash
cd frontend_dashboard
pip install -r requirements.txt
```

## Point to your data

Edit `data/loader.py` — update `BASE_DIR` to point to the folder where your notebooks save outputs:

```python
BASE_DIR = Path("/path/to/your/project")   # folder containing:
                                           # restaurants_agg_performance.parquet
                                           # activity_performance_with_roi.csv
                                           # priority_list.csv
```

## Run

```bash
streamlit run app.py
```

## Pages

| Page | What it shows |
|---|---|
| 📊 Overview | Restaurant explorer — booking volume, revenue, growth charts, full portfolio table |
| 📈 Momentum | Segment distribution, strategic matrix, YoY vs MoM signal breakdown, growth heatmap |
| 🎯 Priority List | Ranked 251 stable-growth restaurants with tier labels, filters, and channel recommendations |
| 🤖 Strategy Engine | Per-restaurant AI marketing strategy brief powered by Claude API |

## Data freshness

The loader caches data for 5 minutes (`@st.cache_data(ttl=300)`).
Re-run your notebooks → the dashboard picks up new data within 5 minutes automatically.
To force an immediate refresh, press **C** in the Streamlit app to clear cache.

## Demo mode

If notebook output files are not found, the app runs on realistic sample data
and shows a warning banner. Everything is functional — swap in real files when ready.

## Project structure

```
streamlit_app/
├── app.py                  ← entry point + global CSS + navigation
├── requirements.txt
├── data/
│   └── loader.py           ← reads parquet/csv, generates sample data for dev
└── pages/
    ├── overview.py         ← Page 1: Restaurant explorer
    ├── momentum.py         ← Page 2: Momentum & segment dashboard
    ├── priority.py         ← Page 3: Ranked priority list
    └── strategy.py         ← Page 4: Claude API strategy engine
```
