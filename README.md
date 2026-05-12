# OPE

This repo is now trimmed to the active dashboard workflow plus the minimum set of notebooks and outputs that feed it.

## Start Here
- Dashboard entrypoint: `frontend_dashboard/app.py`
- Dashboard data layer: `frontend_dashboard/data/loader.py`

## Pipeline Structure

### `frontend_dashboard/`
- Active Streamlit app.

### `_1_eda/`
- Notebook: `main.ipynb`
- Output folder: `data_output/`
- Kept output: `data_output/places_api_new_results.csv`
- Purpose: base restaurant enrichment used by clustering.

### `_2_feature_engineering+momentum/`
- Notebook: `momentum_seasonality_updated.ipynb`
- Output folder: `data_output/`
- Kept files:
  - `data_output/bookings_cleaned.parquet`
  - `data_output/valid_bookings_for_marketing.parquet`
  - `data_output/restaurants_agg_performance.parquet`
  - `data_output/priority_latest_momentum_labels.parquet`
- Purpose: momentum features and restaurant-month performance.

### `_3_marketing/`
- Notebooks:
  - `aggregation.ipynb`
  - `compare_bookings_with_marketing_new.ipynb`
- Output folder: `data_output/`
- Kept files:
  - `data_output/master_marketing_activties.csv`
  - `data_output/activity_performance_with_roi.csv`
  - `data_output/restaurant_reviews.parquet`
- Purpose: master marketing table, marketing lift/ROI, and review corpus used by clustering.

### `_4_final_outputs/`
- Notebook: `priority_scoring_seasonality.ipynb`
- Output folder: `data_output/`
- Kept output: `data_output/priority_list.csv`
- Purpose: final priority ranking.

### `_5_GA_data/`
- Notebook: `ga_campaign_alignment.ipynb`
- Output folder: `data_output/`
- Kept scripts:
  - `googleapi_clean.py`
  - `googleapi_create_datasets.py`
  - `gmv_datasets_build.py`
- Kept outputs:
  - `data_output/combined_restaurant_ga.parquet`
  - `data_output/gmv/gmv_view.parquet`
  - `data_output/gmv/gmv_monthly.parquet`
- Purpose: GA alignment and GMV-per-GA-view enrichment.

### `_6_web_scraping/`
- Notebooks:
  - `pull_cities.ipynb`
  - `pull_names.ipynb`
- Purpose: Scrape location for unique restaurant names from booking file

### `clustering/`
- Notebook: `clustering.ipynb`
- Output folder: `data_output/`
- Kept outputs:
  - `data_output/reviews.csv`
  - `data_output/clustering_results.csv`
- Purpose: restaurant text clustering used by the clustering page.

### `data/`
- Raw source data only.

### `scripts/`
- `clean_notebooks.py` standardizes the remaining notebooks and keeps them output-free in git.

## Run The Dashboard

```bash
cd frontend_dashboard
streamlit run app.py
```

### Environment set up
- COHERE_API_KEY