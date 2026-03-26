# Clustering

This folder is reduced to the files the dashboard actually uses.

## Kept Files

- `clustering.ipynb`
  - Generates the clustering outputs.
- `data_output/reviews.csv`
  - Review-level text corpus with cluster labels.
- `data_output/clustering_results.csv`
  - Restaurant-level cluster assignment with 2D coordinates.

## Inputs Used By The Notebook

- `../data/kol/KOL_Posts.csv`
- `../_1_eda/data_output/places_api_new_results.csv`
- `../_3_marketing/data_output/restaurant_reviews.parquet`
