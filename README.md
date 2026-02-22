# OPE Project

## Overview
This project contains exploratory data analysis (EDA), feature engineering, marketing analytics, and clustering analysis for restaurant booking data.

## Folder Structure

### `_1_eda/` - Exploratory Data Analysis

This folder contains scripts and notebooks for initial data exploration, cleaning, and preparation:

- **`create_datasets_googleapi.py`** - Fetches data from Google Analytics using the GA4 API. Retrieves and saves dimension and metric metadata to CSV files in `data/marketing/googleAPI/metadata/`.

- **`facebook_ads.ipynb`** - Cleans and processes Facebook Ads campaign data. Parses ad names to extract features, standardizes columns, handles date conversions, and filters out invalid restaurant IDs.

- **`geolocation_cuisine_added.py`** - Enriches restaurant data with cuisine information. Uses Google Places API data to infer cuisine types based on place types (e.g., "japanese_restaurant" → "Japanese"). Implements a weighted scoring system for multiple cuisine classifications.

- **`geolocation_restaurants.py`** - Extracts standardized location data from Google Places API. Parses address components to identify city, country, and postal code with fallback logic for different address formats.

- **`kol_booking_data_cleaning.ipynb`** - Cleans Key Opinion Leader (KOL) booking data from Singapore and Thailand markets. Standardizes column names across regions, concatenates data, handles follower count formatting, and generates marketing summary features.

- **`kol_posts_data_cleaning.ipynb`** - Cleans KOL social media post data. Processes metrics including views, likes, comments, and posting dates. Handles numeric field cleaning and data type conversions.

- **`main.ipynb`** - Main comprehensive EDA notebook. Performs data loading, exploratory analysis, missing data analysis, outlier detection using statistical methods, data visualization, feature engineering for growth and trend metrics, and city/cuisine hierarchy extraction.

- **`places_api_new_results.csv`** - Output file containing processed results from Google Places API queries with restaurant location and attribute data.

### `_2_feature_engineering+momentum/` - Feature Engineering and Growth Momentum Analysis

This folder contains advanced analytics for identifying restaurant growth potential and momentum scoring:

#### `claude/` - Configurable Growth Analysis System

- **`restaurant_growth_analysis.py`** - Core module for restaurant growth momentum analysis. Calculates month-over-month and rolling average growth rates, segments restaurants into actionable categories (Rising Stars, Emerging Opportunities, Established Players, Needs Attention), and generates momentum scores based on volume and growth trends.

- **`restaurant_growth_analysis_adjustable.py`** - Configurable version of growth analysis with adjustable parameters. Allows customization of time periods, volume thresholds, growth thresholds, momentum score weights, and segmentation methods through the `GrowthAnalysisConfig` class.

- **`growth_analysis_notebook.ipynb`** - Interactive Jupyter notebook for running growth momentum analysis. Demonstrates usage of the analysis modules, visualizes growth patterns, generates momentum matrices, and creates executive summary reports with restaurant rankings.

- **`QUICK_START_GUIDE.md`** - Comprehensive documentation explaining the growth momentum methodology, segment definitions, marketing budget allocation recommendations (50% to Rising Stars, 30% to Emerging Opportunities, 15% to Established Players, 5% to Needs Attention), and key differences from static performance matrices.

- **`analysis_config.txt`** - Configuration file storing analysis parameters including recent months window, volume/growth thresholds, momentum score weights, and segmentation method preferences.

#### `start/` - Initial Implementation

- **`momentum.py`** - Original momentum feature engineering script. Builds monthly restaurant aggregations, calculates growth metrics (booking growth, revenue growth), generates rolling averages, handles infinity values from zero-to-nonzero growth, and creates time-based features for trend analysis.

- **`momentum.ipynb`** - Notebook version of momentum engineering with visual checks at each step. Mirrors the `momentum.py` script functionality while providing data quality validation, distribution analysis, and intermediate visualizations of growth patterns.

### `clustering/` - Restaurant Theme Segmentation via Text Analysis

This folder contains customer perception analysis using text mining and machine learning clustering:

- **`clustering.ipynb`** - Main clustering notebook that segments restaurants by customer perception themes. Aggregates text data from multiple sources (KOL posts, restaurant reviews, Google Places data), applies TF-IDF vectorization to convert text to numeric features, uses K-Means clustering (7 clusters) to identify themes, assigns restaurants to predefined themes like "Various Menus & Quality Food", "Restaurant with Good View", "Attentive Service but Pricey", "Good Location", and "Extensive Alcohol Beverage Options". Implements multi-theme assignment (restaurants can belong to multiple themes if they meet threshold criteria), uses UMAP dimensionality reduction for 2D visualization, and generates interactive Plotly scatter plots showing restaurant positioning in theme space.