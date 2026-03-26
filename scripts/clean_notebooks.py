from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def to_source(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(text),
    }


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def rewrite_compare_marketing_new() -> None:
    path = ROOT / "_3_marketing" / "compare_bookings_with_marketing_new.ipynb"
    if not path.exists():
        return

    nb = load_nb(path)
    nb["cells"] = [
        md_cell(
            """
            # Marketing Attribution and ROI

            Generates the final marketing effectiveness output
            `data_output/activity_performance_with_roi.csv` from
            `../_2_feature_engineering+momentum/data_output/valid_bookings_for_marketing.parquet`
            and `data_output/master_marketing_activties.csv`.
            """
        ),
        code_cell(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from pathlib import Path

            BASE_DIR = Path.cwd().parent
            BOOKINGS_PATH = BASE_DIR / "_2_feature_engineering+momentum" / "data_output" / "valid_bookings_for_marketing.parquet"
            MARKETING_PATH = Path("data_output") / "master_marketing_activties.csv"
            OUT_DIR = Path("data_output")
            OUT_DIR.mkdir(parents=True, exist_ok=True)

            master_bookings = pd.read_parquet(BOOKINGS_PATH)
            marketing_activities = pd.read_csv(MARKETING_PATH)
            """
        ),
        md_cell("## 1. Attribute Bookings to Marketing Windows"),
        code_cell(
            """
            bookings = master_bookings.copy()
            if "booking_date" in bookings.columns:
                bookings["booking_datetime"] = pd.to_datetime(bookings["booking_date"], errors="coerce")
            else:
                bookings["booking_datetime"] = pd.to_datetime(bookings["date"], errors="coerce")

            if "revenue_thb" not in bookings.columns and "revenue" in bookings.columns:
                bookings["revenue_thb"] = pd.to_numeric(bookings["revenue"], errors="coerce")
            if "total_guests" not in bookings.columns and "party_size" in bookings.columns:
                bookings["total_guests"] = bookings["party_size"]

            bookings["restaurant_id"] = pd.to_numeric(bookings["restaurant_id"], errors="coerce")
            bookings = bookings.dropna(subset=["restaurant_id", "booking_datetime"]).copy()

            mkt = marketing_activities.copy()
            mkt["restaurant_id"] = (
                pd.to_numeric(mkt.get("crm_restaurant_id"), errors="coerce")
                .combine_first(pd.to_numeric(mkt.get("kol_restaurant_id"), errors="coerce"))
                .combine_first(pd.to_numeric(mkt.get("fb_restaurant_id"), errors="coerce"))
            )
            mkt["activity_start"] = pd.to_datetime(mkt["activity_start"], errors="coerce").dt.normalize()
            mkt["activity_end"] = pd.to_datetime(mkt["activity_end"], errors="coerce").dt.normalize()
            mkt["activity_end"] = mkt["activity_end"].fillna(
                mkt["activity_start"] + pd.to_timedelta(pd.to_numeric(mkt.get("fb_campaign_duration_days"), errors="coerce"), unit="D")
            )
            mkt["activity_end"] = mkt["activity_end"].fillna(pd.Timestamp.today().normalize())
            mkt = mkt.dropna(subset=["activity_id", "restaurant_id", "activity_start", "activity_end"]).copy()
            mkt = mkt.loc[mkt["activity_end"] >= mkt["activity_start"]].copy()

            mkt_exposure = mkt[[
                "activity_id", "channel", "restaurant_id",
                "activity_start", "activity_end",
                "crm_campaign_name", "crm_topic", "crm_audience",
                "kol_platform", "kol_username", "kol_post_url",
                "fb_campaign", "fb_amount_spent_thb",
            ]].copy()

            cand = bookings[["id", "restaurant_id", "booking_datetime", "revenue_thb", "total_guests"]].merge(
                mkt_exposure,
                on="restaurant_id",
                how="left",
                suffixes=("_booking", "_mkt"),
            )
            attrib = cand[
                (cand["booking_datetime"] >= cand["activity_start"])
                & (cand["booking_datetime"] <= cand["activity_end"])
            ].copy()
            attrib["time_from_start_hours"] = (attrib["booking_datetime"] - attrib["activity_start"]).dt.total_seconds() / 3600.0
            attrib["abs_time_from_start_hours"] = attrib["time_from_start_hours"].abs()
            attrib = attrib.sort_values(["id", "abs_time_from_start_hours"])
            attrib_1 = attrib.drop_duplicates(subset=["id"], keep="first").copy()

            print("Bookings:", bookings.shape)
            print("Exposures:", mkt_exposure.shape)
            print("Attributed bookings:", attrib_1.shape)
            attrib_1.head()
            """
        ),
        md_cell("## 2. Calculate Lift per Activity"),
        code_cell(
            """
            mkt_lift = mkt_exposure.copy()
            mkt_lift["window_hours"] = (mkt_lift["activity_end"] - mkt_lift["activity_start"]).dt.total_seconds() / 3600.0
            mkt_lift = mkt_lift.loc[mkt_lift["window_hours"] > 0].copy()
            mkt_lift["baseline_start"] = mkt_lift["activity_start"] - pd.to_timedelta(mkt_lift["window_hours"], unit="h")
            mkt_lift["baseline_end"] = mkt_lift["activity_start"]

            cand_during = bookings[["id", "restaurant_id", "booking_datetime"]].merge(
                mkt_lift[["activity_id", "restaurant_id", "activity_start", "activity_end"]],
                on="restaurant_id",
                how="inner",
            )
            during_counts = cand_during[
                (cand_during["booking_datetime"] >= cand_during["activity_start"])
                & (cand_during["booking_datetime"] <= cand_during["activity_end"])
            ].groupby("activity_id").size().rename("bookings_during")

            cand_baseline = bookings[["id", "restaurant_id", "booking_datetime"]].merge(
                mkt_lift[["activity_id", "restaurant_id", "baseline_start", "baseline_end"]],
                on="restaurant_id",
                how="left",
            )
            baseline = cand_baseline[
                (cand_baseline["booking_datetime"] >= cand_baseline["baseline_start"])
                & (cand_baseline["booking_datetime"] < cand_baseline["baseline_end"])
            ].copy()
            baseline_counts = baseline.groupby("activity_id").size().rename("bookings_baseline")

            lift_table = pd.concat([during_counts, baseline_counts], axis=1).fillna(0).reset_index()
            lift_table["lift"] = lift_table["bookings_during"] - lift_table["bookings_baseline"]

            mkt_meta = mkt_lift[[
                "activity_id", "channel", "restaurant_id", "activity_start", "activity_end", "window_hours",
                "crm_campaign_name", "crm_topic", "crm_audience",
                "kol_platform", "kol_username", "kol_post_url",
                "fb_campaign", "fb_amount_spent_thb",
            ]].drop_duplicates("activity_id")

            activity_perf = lift_table.merge(mkt_meta, on="activity_id", how="left")
            activity_perf["window_days"] = activity_perf["window_hours"] / 24.0
            activity_perf["lift_per_day"] = activity_perf["lift"] / activity_perf["window_days"]
            activity_perf["fb_amount_spent_thb"] = pd.to_numeric(activity_perf["fb_amount_spent_thb"], errors="coerce")
            activity_perf["cost_per_incremental_booking"] = np.where(
                (activity_perf["channel"] == "FB")
                & activity_perf["lift"].gt(0)
                & activity_perf["fb_amount_spent_thb"].gt(0),
                activity_perf["fb_amount_spent_thb"] / activity_perf["lift"],
                np.nan,
            )

            activity_perf.head()
            """
        ),
        md_cell("## 3. Channel Summary"),
        code_cell(
            """
            channel_summary = (
                activity_perf
                .groupby("channel", as_index=False)
                .agg(
                    activities=("activity_id", "nunique"),
                    total_bookings_during=("bookings_during", "sum"),
                    total_baseline=("bookings_baseline", "sum"),
                    total_lift=("lift", "sum"),
                    avg_lift=("lift", "mean"),
                    median_lift=("lift", "median"),
                )
            )
            channel_summary["lift_rate_vs_baseline"] = np.where(
                channel_summary["total_baseline"] > 0,
                channel_summary["total_lift"] / channel_summary["total_baseline"],
                np.nan,
            )
            channel_summary
            """
        ),
        md_cell("## 4. Incremental Revenue and ROI"),
        code_cell(
            """
            revenue_data = attrib_1.groupby("activity_id").agg(total_campaign_revenue=("revenue_thb", "sum")).reset_index()
            activity_perf = activity_perf.merge(revenue_data, on="activity_id", how="left")
            activity_perf["total_campaign_revenue"] = activity_perf["total_campaign_revenue"].fillna(0)
            activity_perf["aov_thb"] = np.where(
                activity_perf["bookings_during"] > 0,
                activity_perf["total_campaign_revenue"] / activity_perf["bookings_during"],
                0,
            )
            activity_perf["incremental_revenue_thb"] = activity_perf["lift"] * activity_perf["aov_thb"]
            activity_perf["fb_amount_spent_thb"] = pd.to_numeric(activity_perf["fb_amount_spent_thb"], errors="coerce").fillna(0)
            activity_perf["roi"] = np.where(
                (activity_perf["channel"] == "FB") & (activity_perf["fb_amount_spent_thb"] > 0),
                (activity_perf["incremental_revenue_thb"] - activity_perf["fb_amount_spent_thb"]) / activity_perf["fb_amount_spent_thb"],
                np.nan,
            )
            activity_perf["roi_percentage"] = activity_perf["roi"] * 100
            activity_perf.to_csv(OUT_DIR / "activity_performance_with_roi.csv", index=False)
            activity_perf.sort_values("roi", ascending=False).head(10)
            """
        ),
        md_cell("## 5. Diagnostic Charts"),
        code_cell(
            """
            plot_data = activity_perf.dropna(subset=["roi"]).copy()
            if not plot_data.empty:
                top_roi = plot_data.sort_values("roi", ascending=False).head(10)
                top_revenue = plot_data.sort_values("incremental_revenue_thb", ascending=False).head(10)
                plt.style.use("seaborn-v0_8-whitegrid")

                plt.figure(figsize=(10, 6))
                bars1 = plt.barh(top_roi["activity_id"].astype(str), top_roi["roi_percentage"], color="#4C72B0")
                plt.title("Top 10 Campaigns by Marketing ROI (%)")
                plt.xlabel("Return on Investment (%)")
                plt.ylabel("Campaign (Activity ID)")
                plt.gca().invert_yaxis()
                for bar in bars1:
                    width = bar.get_width()
                    plt.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2, f"{width:.0f}%", va="center", fontsize=10)
                plt.axvline(0, color="black", linewidth=1)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(10, 6))
                bars2 = plt.barh(top_revenue["activity_id"].astype(str), top_revenue["incremental_revenue_thb"], color="#55A868")
                plt.title("Top 10 Campaigns by Incremental Revenue (THB)")
                plt.xlabel("Incremental Revenue Generated (THB)")
                plt.ylabel("Campaign (Activity ID)")
                plt.gca().invert_yaxis()
                for bar in bars2:
                    width = bar.get_width()
                    plt.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2, f"฿{width:,.0f}", va="center", fontsize=10)
                plt.tight_layout()
                plt.show()
            else:
                print("No ROI rows available for plotting.")
            """
        ),
    ]
    save_nb(path, nb)


def rewrite_ga_campaign_alignment() -> None:
    path = ROOT / "_5_GA_data" / "ga_campaign_alignment.ipynb"
    if not path.exists():
        return

    nb = load_nb(path)
    nb["cells"] = [
        md_cell(
            """
            # Google Analytics Campaign Alignment

            Builds two outputs:
            - `monthly_gmv_per_ga_view`: monthly portfolio-level GMV per GA view
            - `data_output/combined_restaurant_ga.parquet`: restaurant-month momentum data enriched with GA view metrics
            """
        ),
        code_cell(
            """
            from pathlib import Path

            import pandas as pd

            BASE_DIR = Path("..")
            RESTAURANTS_PATH = BASE_DIR / "_2_feature_engineering+momentum" / "data_output" / "restaurants_agg_performance.parquet"
            GMV_VIEW_PATH = BASE_DIR / "_5_GA_data" / "data_output" / "gmv" / "gmv_view.parquet"
            OUTPUT_DIR = Path("data_output")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            COMBINED_OUT = OUTPUT_DIR / "combined_restaurant_ga.parquet"

            res = pd.read_parquet(RESTAURANTS_PATH)
            gmv_view_df = pd.read_parquet(GMV_VIEW_PATH)

            def _safe_divide(numerator, denominator):
                return pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce").replace(0, pd.NA)
            """
        ),
        md_cell("## Validate Restaurant-Month Grain"),
        code_cell(
            """
            gmv_view = gmv_view_df.copy()
            gmv_view["restaurant_id"] = pd.to_numeric(gmv_view["restaurant_id"], errors="coerce").astype("Int64")
            gmv_view["year_month"] = pd.to_datetime(gmv_view["year_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()

            duplicate_keys = int(gmv_view.duplicated(["restaurant_id", "year_month"]).sum())
            print("gmv_view rows:", len(gmv_view))
            print("duplicate restaurant-month rows:", duplicate_keys)
            print("restaurants with GA data:", gmv_view["restaurant_id"].nunique())
            print("months with GA data:", gmv_view["year_month"].nunique())

            if duplicate_keys:
                gmv_view.loc[
                    gmv_view.duplicated(["restaurant_id", "year_month"], keep=False),
                    ["restaurant_id", "name", "year_month", "ga_items_viewed", "monthly_gmv"],
                ].sort_values(["restaurant_id", "year_month"]).head(20)
            else:
                gmv_view[["restaurant_id", "name", "year_month", "ga_items_viewed", "monthly_gmv", "gmv_per_view"]].head()
            """
        ),
        md_cell("## Monthly GMV per GA View"),
        code_cell(
            """
            restaurant_ga_monthly = (
                gmv_view_df.copy()
                .assign(
                    restaurant_id=lambda df: pd.to_numeric(df["restaurant_id"], errors="coerce").astype("Int64"),
                    year_month=lambda df: pd.to_datetime(df["year_month"], errors="coerce").dt.to_period("M").dt.to_timestamp(),
                )
                .sort_values(["restaurant_id", "year_month"])
                .reset_index(drop=True)
            )
            restaurant_ga_monthly["gmv_per_ga_view"] = restaurant_ga_monthly["gmv_per_view"]
            restaurant_ga_monthly["bookings_per_ga_view"] = _safe_divide(
                restaurant_ga_monthly["monthly_bookings"],
                restaurant_ga_monthly["ga_items_viewed"],
            )
            restaurant_ga_monthly["ga_view_to_purchase_rate"] = restaurant_ga_monthly["view_to_purchase_rate"]

            monthly_gmv_per_ga_view = (
                restaurant_ga_monthly
                .groupby("year_month", as_index=False)
                .agg(
                    restaurants_with_ga_data=("restaurant_id", "nunique"),
                    total_monthly_gmv=("monthly_gmv", "sum"),
                    total_monthly_bookings=("monthly_bookings", "sum"),
                    total_ga_items_viewed=("ga_items_viewed", "sum"),
                    total_ga_items_added_to_cart=("ga_items_added_to_cart", "sum"),
                    total_ga_items_purchased=("ga_items_purchased", "sum"),
                    total_ga_item_revenue=("ga_item_revenue", "sum"),
                )
                .sort_values("year_month")
            )

            monthly_gmv_per_ga_view["gmv_per_ga_view"] = _safe_divide(monthly_gmv_per_ga_view["total_monthly_gmv"], monthly_gmv_per_ga_view["total_ga_items_viewed"])
            monthly_gmv_per_ga_view["bookings_per_ga_view"] = _safe_divide(monthly_gmv_per_ga_view["total_monthly_bookings"], monthly_gmv_per_ga_view["total_ga_items_viewed"])
            monthly_gmv_per_ga_view["ga_view_to_purchase_rate"] = _safe_divide(monthly_gmv_per_ga_view["total_ga_items_purchased"], monthly_gmv_per_ga_view["total_ga_items_viewed"])

            monthly_gmv_per_ga_view
            """
        ),
        md_cell("## Restaurant-Month Combined Output"),
        code_cell(
            """
            restaurant_perf = res.copy()
            restaurant_perf["restaurant_id"] = pd.to_numeric(restaurant_perf["restaurant_id"], errors="coerce").astype("Int64")
            restaurant_perf["year_month"] = pd.to_datetime(restaurant_perf["year_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()

            ga_monthly = (
                gmv_view_df[
                    [
                        "restaurant_id",
                        "year_month",
                        "ga_items_viewed",
                        "ga_items_added_to_cart",
                        "ga_items_purchased",
                        "ga_item_revenue",
                        "gmv_per_view",
                        "bookings_per_view",
                        "ga_add_to_cart_rate",
                        "view_to_purchase_rate",
                        "purchase_to_cart_rate",
                        "revenue_per_view",
                    ]
                ]
                .copy()
                .assign(
                    restaurant_id=lambda df: pd.to_numeric(df["restaurant_id"], errors="coerce").astype("Int64"),
                    year_month=lambda df: pd.to_datetime(df["year_month"], errors="coerce").dt.to_period("M").dt.to_timestamp(),
                    gmv_per_ga_view=lambda df: df["gmv_per_view"],
                    bookings_per_ga_view=lambda df: df["bookings_per_view"],
                    ga_view_to_purchase_rate=lambda df: df["view_to_purchase_rate"],
                    ga_purchase_to_cart_rate=lambda df: df["purchase_to_cart_rate"],
                    ga_revenue_per_view=lambda df: df["revenue_per_view"],
                )
            )

            duplicate_keys = int(ga_monthly.duplicated(["restaurant_id", "year_month"]).sum())
            if duplicate_keys:
                raise ValueError(f"gmv_view.parquet is not unique at restaurant-month grain: {duplicate_keys} duplicate keys")

            combined = restaurant_perf.merge(
                ga_monthly,
                on=["restaurant_id", "year_month"],
                how="left",
                validate="one_to_one",
            )

            combined.to_parquet(COMBINED_OUT, index=False)
            print(f"Saved: {COMBINED_OUT}")
            print(combined.shape)
            print("rows with GA data:", int(combined["ga_items_viewed"].fillna(0).gt(0).sum()))
            combined.head()
            """
        ),
    ]
    save_nb(path, nb)

def fix_clustering_notebook() -> None:
    path = ROOT / "clustering" / "clustering.ipynb"
    if not path.exists():
        return

    nb = load_nb(path)
    nb["cells"] = [
        md_cell(
            """
            # Restaurant Clustering

            Builds the clustering outputs used by the dashboard:
            - `data_output/reviews.csv`: review-level text corpus with cluster labels
            - `data_output/clustering_results.csv`: restaurant-level cluster assignments with 2D coordinates
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import re

            import numpy as np
            import pandas as pd
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            from umap import UMAP

            BASE_DIR = Path("..")
            KOL_PATH = BASE_DIR / "data" / "kol" / "KOL_Posts.csv"
            PLACES_PATH = BASE_DIR / "_1_eda" / "data_output" / "places_api_new_results.csv"
            REVIEWS_PATH = BASE_DIR / "_3_marketing" / "data_output" / "restaurant_reviews.parquet"
            OUTPUT_DIR = Path("data_output")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            kol_posts = pd.read_csv(KOL_PATH)
            places = pd.read_csv(PLACES_PATH)
            restaurant_reviews = pd.read_parquet(REVIEWS_PATH)
            """
        ),
        md_cell("## 1. Build the Restaurant Text Corpus"),
        code_cell(
            """
            kol_aggregated = (
                kol_posts.groupby("Restaurant Name")["Content"]
                .apply(lambda s: " ".join(s.dropna().astype(str)))
                .reset_index(name="kol_text")
            )

            places = places.copy()
            places["google_text"] = (
                places["input_string"].fillna("").astype(str)
                + " "
                + places["raw_types"].fillna("").astype(str).str.replace(",", " ")
                + " "
                + places["Cuisine"].fillna("").astype(str)
            )

            place_text = (
                places[["input_string", "google_text"]]
                .rename(columns={"input_string": "restaurant name"})
                .drop_duplicates("restaurant name")
            )

            review_text = (
                restaurant_reviews.groupby("input_restaurant_name")["review_text"]
                .apply(lambda s: " ".join(s.dropna().astype(str)))
                .reset_index(name="review_text")
                .rename(columns={"input_restaurant_name": "restaurant name"})
            )

            reviews = (
                review_text
                .merge(place_text, on="restaurant name", how="left")
                .merge(kol_aggregated.rename(columns={"Restaurant Name": "restaurant name"}), on="restaurant name", how="left")
            )
            reviews["raw_text"] = (
                reviews["review_text"].fillna("").astype(str)
                + " "
                + reviews["google_text"].fillna("").astype(str)
                + " "
                + reviews["kol_text"].fillna("").astype(str)
            ).str.replace(r"\\s+", " ", regex=True).str.strip()

            reviews = reviews[reviews["raw_text"].str.len() > 0].copy()
            reviews.head()
            """
        ),
        md_cell("## 2. Clean and Vectorize Text"),
        code_cell(
            """
            redundant_words = [
                "point_of_interest",
                "establishment",
                "general",
                "food establishment",
                "restaurant point_of_interest",
                "food point_of_interest",
                "point_of_interest establishment",
                "establishment general",
                "restaurant food",
                "meal_delivery",
                "meal_takeaway",
                "lodging",
                "tourist_attraction",
                "night_club",
                "shopping_mall",
                "bangkok",
                "sukhumvit",
                "singapore",
                "thai",
                "baht",
                "order",
                "ordered",
                "dishes",
                "fried",
                "sauce",
                "spicy",
                "time",
                "eat",
                "meat",
                "restaurant",
                "soup",
                "super",
                "really",
                "terrible",
                "floor",
                "rice",
                "pork",
                "beef",
                "chicken",
                "crab",
            ]

            def clean_text(text: str) -> str:
                cleaned = str(text).lower()
                for phrase in sorted(redundant_words, key=len, reverse=True):
                    cleaned = cleaned.replace(phrase, " ")
                cleaned = re.sub(r"[^a-z\\s]", " ", cleaned)
                cleaned = re.sub(r"\\s+", " ", cleaned).strip()
                return cleaned

            reviews["clean_text"] = reviews["raw_text"].apply(clean_text)
            reviews = reviews[reviews["clean_text"].str.len() > 0].copy()

            vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
            )
            text_vectors = vectorizer.fit_transform(reviews["clean_text"])
            print(reviews.shape)
            """
        ),
        md_cell("## 3. Cluster Reviews"),
        code_cell(
            """
            cluster_themes = {
                0: "Bar & Alcoholic Drinks",
                1: "Good food and atmosphere",
                2: "Buffet & premium dining",
                3: "Excellent service",
                4: "Cafes, Coffee, Breakfast",
            }

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            reviews["cluster"] = kmeans.fit_predict(text_vectors)
            reviews["theme"] = reviews["cluster"].map(cluster_themes)
            reviews.to_csv(OUTPUT_DIR / "reviews.csv", index=False)

            reviews[["restaurant name", "cluster", "theme"]].head()
            """
        ),
        md_cell("## 4. Build Restaurant-Level Assignments"),
        code_cell(
            """
            reducer = UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=1.0,
                metric="cosine",
                random_state=42,
            )
            embeddings_2d = reducer.fit_transform(text_vectors.toarray())

            review_points = reviews.copy()
            review_points["x"] = embeddings_2d[:, 0]
            review_points["y"] = embeddings_2d[:, 1]

            def dominant_value(series: pd.Series):
                counts = series.value_counts()
                return counts.index[0] if len(counts) else pd.NA

            def dominant_share(series: pd.Series) -> float:
                counts = series.value_counts(normalize=True)
                return float(counts.iloc[0]) if len(counts) else np.nan

            clustering_results = (
                review_points.groupby("restaurant name", as_index=False)
                .agg(
                    cluster=("cluster", dominant_value),
                    theme=("theme", dominant_value),
                    x=("x", "mean"),
                    y=("y", "mean"),
                    cluster_confidence=("cluster", dominant_share),
                    review_count=("raw_text", "size"),
                )
                .sort_values(["cluster", "restaurant name"])
                .reset_index(drop=True)
            )

            clustering_results.to_csv(OUTPUT_DIR / "clustering_results.csv", index=False)
            clustering_results.head()
            """
        ),
    ]
    save_nb(path, nb)


def fix_main_eda_notebook() -> None:
    path = ROOT / "_1_eda" / "main.ipynb"
    if not path.exists():
        return

    nb = load_nb(path)
    if len(nb["cells"]) > 1 and nb["cells"][1].get("cell_type") == "code":
        src = "".join(nb["cells"][1].get("source", [])).strip()
        if "pip install" in src and src.startswith("#"):
            del nb["cells"][1]

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if "places_api_new_results.csv" not in source:
            continue
        source = source.replace("'places_api_new_results.csv'", "'data_output/places_api_new_results.csv'")
        source = source.replace('"places_api_new_results.csv"', '"data_output/places_api_new_results.csv"')
        cell["source"] = to_source(source)
    save_nb(path, nb)


def patch_momentum_output_paths() -> None:
    candidates = [
        ROOT / "_2_feature_engineering+momentum" / "momentum_seasonality_updated.ipynb",
        ROOT / "_2_feature_engineering+momentum" / "start" / "momentum_seasonality_updated.ipynb",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return

    nb = load_nb(path)
    replacements = {
        'BASE_DIR = Path(".")\nmaster_bookings_df = pd.read_parquet(BASE_DIR / "bookings_cleaned.parquet")': (
            'OUTPUT_DIR = Path("..") / "data_output"\n'
            'OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n'
            'master_bookings_df = pd.read_parquet(OUTPUT_DIR / "bookings_cleaned.parquet")'
        ),
        'valid_bookings_df.to_parquet(BASE_DIR / "valid_bookings_for_marketing.parquet", index=False)': (
            'valid_bookings_df.to_parquet(OUTPUT_DIR / "valid_bookings_for_marketing.parquet", index=False)'
        ),
        "analysis_window_df.to_parquet('restaurants_agg_performance.parquet')": (
            'analysis_window_df.to_parquet(OUTPUT_DIR / "restaurants_agg_performance.parquet")'
        ),
        "output_path = Path('.') / 'priority_latest_momentum_labels.parquet'": (
            'output_path = OUTPUT_DIR / "priority_latest_momentum_labels.parquet"'
        ),
    }

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        updated = source
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != source:
            cell["source"] = to_source(updated)

    save_nb(path, nb)


def patch_priority_output_paths() -> None:
    path = ROOT / "_4_final_outputs" / "priority_scoring_seasonality.ipynb"
    if not path.exists():
        return

    nb = load_nb(path)
    replacements = {
        "- `restaurants_agg_performance.parquet` — from `momentum_seasonality.ipynb`": (
            "- `../_2_feature_engineering+momentum/data_output/restaurants_agg_performance.parquet` — from `momentum_seasonality.ipynb`"
        ),
        "- `activity_performance_with_roi.csv` — from `compare_bookings_with_marketing_new.ipynb`": (
            "- `../_3_marketing/data_output/activity_performance_with_roi.csv` — from `compare_bookings_with_marketing_new.ipynb`"
        ),
        "**Output:** `priority_list.csv`": "**Output:** `data_output/priority_list.csv`",
        "MOMENTUM_PATH  = Path('../_2_feature_engineering+momentum/start/restaurants_agg_performance.parquet')": (
            "MOMENTUM_PATH  = Path('../_2_feature_engineering+momentum/data_output/restaurants_agg_performance.parquet')"
        ),
        "MARKETING_PATH = Path('../_3_marketing/activity_performance_with_roi.csv')": (
            "MARKETING_PATH = Path('../_3_marketing/data_output/activity_performance_with_roi.csv')"
        ),
        "OUTPUT_PATH    = Path('priority_list.csv')": (
            'OUTPUT_DIR = Path("data_output")\n'
            'OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n'
            'OUTPUT_PATH    = OUTPUT_DIR / "priority_list.csv"'
        ),
        "SEGMENT_OUTPUT_PATH = Path('../_2_feature_engineering+momentum/start/priority_latest_momentum_labels.parquet')": (
            "SEGMENT_OUTPUT_PATH = Path('../_2_feature_engineering+momentum/data_output/priority_latest_momentum_labels.parquet')"
        ),
    }

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        updated = source
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != source:
            cell["source"] = to_source(updated)

    save_nb(path, nb)


def patch_marketing_output_paths() -> None:
    path = ROOT / "_3_marketing" / "aggregation.ipynb"
    if not path.exists():
        return

    nb = load_nb(path)
    replacements = {
        'OUT_DIR = BASE_DIR / "_3_marketing"': (
            'OUT_DIR = BASE_DIR / "_3_marketing" / "data_output"\n'
            "OUT_DIR.mkdir(parents=True, exist_ok=True)"
        ),
    }

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        updated = source
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != source:
            cell["source"] = to_source(updated)

    save_nb(path, nb)


def global_cleanup() -> None:
    notebooks = [p for p in sorted(ROOT.rglob("*.ipynb")) if p.exists() and ".ipynb_checkpoints" not in str(p)]
    for path in notebooks:
        nb = load_nb(path)
        cleaned_cells = []
        for cell in nb.get("cells", []):
            cell["metadata"] = {k: v for k, v in cell.get("metadata", {}).items() if k == "tags"}
            cell["source"] = to_source("".join(cell.get("source", [])))
            source_text = "".join(cell.get("source", [])).strip()
            if cell.get("cell_type") == "code":
                if not source_text:
                    continue
                if source_text.startswith("Traceback (most recent call last):"):
                    continue
                if source_text.startswith("UnicodeEncodeError:"):
                    continue
                cell["execution_count"] = None
                cell["outputs"] = []
            cleaned_cells.append(cell)
        nb["cells"] = cleaned_cells

        if not any(c.get("cell_type") == "markdown" for c in nb["cells"]):
            title = path.stem.replace("_", " ").replace("-", " ").title()
            nb["cells"].insert(0, md_cell(f"# {title}"))

        top_meta = nb.get("metadata", {})
        nb["metadata"] = {k: v for k, v in top_meta.items() if k in {"kernelspec", "language_info"}}
        save_nb(path, nb)

    print(f"Cleaned notebooks: {len(notebooks)}")


def main() -> None:
    rewrite_compare_marketing_new()
    rewrite_ga_campaign_alignment()
    fix_clustering_notebook()
    fix_main_eda_notebook()
    patch_momentum_output_paths()
    patch_marketing_output_paths()
    patch_priority_output_paths()
    global_cleanup()


if __name__ == "__main__":
    main()
