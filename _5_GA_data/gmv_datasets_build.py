from pathlib import Path

import numpy as np
import pandas as pd

import googleapi_clean


def _normalize_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.casefold()
        .str.replace(r"[^\w\s]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _safe_divide(numerator, denominator) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator,
        errors="coerce",
    ).replace(0, np.nan)


def build_gmv_datasets(
    ga_ci,
    ga_co,
    agg_bk,
    output_dir: str = "../_5_GA_data/data_output/gmv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci = ga_ci.copy()
    ci["year_month"] = pd.to_datetime(ci["yearMonth"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    ci["name_clean"] = _normalize_name(ci["itemName"])
    ci = ci[ci["name_clean"].ne("")].copy()

    restaurant_lookup = (
        agg_bk[["restaurant_id", "name"]]
        .drop_duplicates()
        .assign(name_clean=lambda df: _normalize_name(df["name"]))
    )
    duplicate_name_map = restaurant_lookup[restaurant_lookup.duplicated("name_clean", keep=False)]
    if not duplicate_name_map.empty:
        duplicate_keys = duplicate_name_map["name_clean"].drop_duplicates().tolist()[:10]
        raise ValueError(
            "Normalized restaurant names are not unique in agg_bk. "
            f"Example conflicting keys: {duplicate_keys}"
        )

    ci_monthly = (
        ci.groupby(["name_clean", "year_month"], as_index=False)
        .agg(
            ga_items_viewed=("itemsViewed", "sum"),
            ga_items_added_to_cart=("itemsAddedToCart", "sum"),
            ga_items_purchased=("itemsPurchased", "sum"),
            ga_item_revenue=("itemRevenue", "sum"),
        )
    )
    ci_restaurant_monthly = ci_monthly.merge(
        restaurant_lookup[["restaurant_id", "name", "name_clean"]],
        on="name_clean",
        how="inner",
        validate="many_to_one",
    )

    bk = agg_bk.copy()
    bk["year_month"] = pd.to_datetime(bk["year_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    gmv_view = ci_restaurant_monthly.merge(
        bk[["restaurant_id", "year_month", "monthly_gmv", "monthly_bookings"]],
        on=["restaurant_id", "year_month"],
        how="inner",
        validate="one_to_one",
    )
    gmv_view["yearMonth"] = gmv_view["year_month"].dt.strftime("%Y%m")
    gmv_view["gmv_per_view"] = _safe_divide(gmv_view["monthly_gmv"], gmv_view["ga_items_viewed"])
    gmv_view["bookings_per_view"] = _safe_divide(gmv_view["monthly_bookings"], gmv_view["ga_items_viewed"])
    gmv_view["ga_add_to_cart_rate"] = _safe_divide(
        gmv_view["ga_items_added_to_cart"],
        gmv_view["ga_items_viewed"],
    )
    gmv_view["view_to_purchase_rate"] = _safe_divide(
        gmv_view["ga_items_purchased"],
        gmv_view["ga_items_viewed"],
    )
    gmv_view["purchase_to_cart_rate"] = _safe_divide(
        gmv_view["ga_items_purchased"],
        gmv_view["ga_items_added_to_cart"],
    )
    gmv_view["revenue_per_view"] = _safe_divide(
        gmv_view["ga_item_revenue"],
        gmv_view["ga_items_viewed"],
    )
    gmv_view = gmv_view.drop(columns=["name_clean"]).sort_values(["restaurant_id", "year_month"]).reset_index(drop=True)

    camp_monthly = (
        ga_co.copy()
        .assign(year_month=lambda df: pd.to_datetime(df["yearMonth"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .groupby("year_month", as_index=False)
        .agg(
            total_campaign_sessions=("sessions", "sum"),
            active_campaigns=("campaignId", "nunique"),
        )
    )

    gmv_monthly = (
        gmv_view.groupby("year_month", as_index=False)
        .agg(
            restaurants_with_ga_data=("restaurant_id", "nunique"),
            total_gmv=("monthly_gmv", "sum"),
            total_bookings=("monthly_bookings", "sum"),
            total_views=("ga_items_viewed", "sum"),
            total_add_to_cart=("ga_items_added_to_cart", "sum"),
            total_purchases=("ga_items_purchased", "sum"),
            total_ga_revenue=("ga_item_revenue", "sum"),
        )
        .merge(camp_monthly, on="year_month", how="left")
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    gmv_monthly["yearMonth"] = gmv_monthly["year_month"].dt.strftime("%Y%m")
    gmv_monthly["gmv_per_view"] = _safe_divide(gmv_monthly["total_gmv"], gmv_monthly["total_views"])
    gmv_monthly["bookings_per_view"] = _safe_divide(gmv_monthly["total_bookings"], gmv_monthly["total_views"])
    gmv_monthly["ga_add_to_cart_rate"] = _safe_divide(
        gmv_monthly["total_add_to_cart"],
        gmv_monthly["total_views"],
    )
    gmv_monthly["view_to_purchase_rate"] = _safe_divide(
        gmv_monthly["total_purchases"],
        gmv_monthly["total_views"],
    )
    gmv_monthly["purchase_to_cart_rate"] = _safe_divide(
        gmv_monthly["total_purchases"],
        gmv_monthly["total_add_to_cart"],
    )
    gmv_monthly["revenue_per_view"] = _safe_divide(
        gmv_monthly["total_ga_revenue"],
        gmv_monthly["total_views"],
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gmv_view_path = out / "gmv_view.parquet"
    gmv_monthly_path = out / "gmv_monthly.parquet"
    gmv_view.to_parquet(gmv_view_path, index=False, engine="pyarrow")
    gmv_monthly.to_parquet(gmv_monthly_path, index=False, engine="pyarrow")

    return gmv_view, gmv_monthly


if __name__ == "__main__":
    base_path = Path("../data/marketing/googleAPI")
    ga_ci = pd.read_parquet(base_path / "campaign_impact.parquet")
    ga_co = pd.read_parquet(base_path / "campaigns_outreach.parquet")
    agg_bk = pd.read_parquet("../_2_feature_engineering+momentum/data_output/restaurants_agg_performance.parquet")

    build_gmv_datasets(
        ga_ci=googleapi_clean.clean(ga_ci, "campaign_impact", verbose=False),
        ga_co=googleapi_clean.clean(ga_co, "campaigns_outreach", verbose=False),
        agg_bk=agg_bk,
    )
