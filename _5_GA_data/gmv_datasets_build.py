import numpy as np
import pandas as pd
from pathlib import Path
import googleapi_clean

def build_gmv_datasets(
    ga_ci,           # campaign_impact GA dataset
    ga_co,           # campaigns_outreach GA dataset
    agg_bk,          # booking dataset
    output_dir: str = '../_5_GA_data/gmv',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # itemId → restaurant_id via name match 
    bridge = (
        ga_ci[['itemId', 'itemName']]          # GA item identifiers
        .drop_duplicates()
        .merge(
            agg_bk[['restaurant_id', 'name']].drop_duplicates(),
            left_on='itemName', right_on='name',         # match on name
            how='inner'
        )[['itemId', 'itemName', 'restaurant_id']]
    )

    # campaign_impact joined with restaurant_id via bridge
    ci = (
        ga_ci.copy()
        .drop(columns=['itemName'], errors='ignore')
        .merge(bridge, on='itemId', how='inner')
    )


    # Align yearMonth format before joining agg_bk
    ci['year_month'] = ci['yearMonth'].astype(str)
    bk = agg_bk[agg_bk['in_analysis_window']].copy()
    bk['year_month'] = bk['year_month'].astype(str)

    gmv_view = (
        ci.merge(
            bk[['restaurant_id', 'year_month', 'monthly_gmv', 'monthly_bookings']],
            on=['restaurant_id', 'year_month'],
            how='inner'
        )
    )
    # Core metrics
    gmv_view['gmv_per_view'] = (
        gmv_view['monthly_gmv'] / gmv_view['itemsViewed'].replace(0, np.nan)
    )
    gmv_view['view_to_purchase_rate'] = (
        gmv_view['itemsPurchased'] / gmv_view['itemsViewed'].replace(0, np.nan)
    )
    gmv_view['revenue_per_view'] = (
        gmv_view['itemRevenue'] / gmv_view['itemsViewed'].replace(0, np.nan)
    )


    camp_monthly = (
        ga_co.copy()
        .groupby('yearMonth')
        .agg(
            total_campaign_sessions = ('sessions',    'sum'),
            active_campaigns        = ('campaignId',  'nunique'),
        )
        .reset_index()
    )
    camp_monthly['year_month'] = camp_monthly['yearMonth'].astype(str)

    gmv_monthly = (
        gmv_view
        .groupby('year_month')
        .agg(
            total_gmv           = ('monthly_gmv',           'sum'),
            total_views         = ('itemsViewed',           'sum'),
            total_bookings      = ('monthly_bookings',      'sum'),
            view_to_purchase    = ('view_to_purchase_rate', 'mean'),
        )
        .reset_index()
    )
    gmv_monthly['gmv_per_view'] = (
        gmv_monthly['total_gmv'] / gmv_monthly['total_views']
    )

    gmv_monthly = (
        gmv_monthly
        .merge(
            camp_monthly[['year_month', 'total_campaign_sessions', 'active_campaigns']],
            on='year_month', how='left'
        )
        .sort_values('year_month')
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gmv_view_path    = out / 'gmv_view.parquet'
    gmv_monthly_path = out / 'gmv_monthly.parquet'
    gmv_view.to_parquet(gmv_view_path,    index=False, engine='pyarrow')
    gmv_monthly.to_parquet(gmv_monthly_path, index=False, engine='pyarrow')

    return gmv_view, gmv_monthly


if __name__ == '__main__':
    BASE_PATH = "../data/marketing/googleAPI/"
    ga_ci = pd.read_parquet(BASE_PATH + "campaign_impact.parquet")
    ga_co = pd.read_parquet(BASE_PATH + "campaigns_outreach.parquet")
    agg_bk = pd.read_parquet("../_2_feature_engineering+momentum/start/restaurants_agg_performance.parquet")

    gmv_view, gmv_monthly = build_gmv_datasets(
        ga_ci          = googleapi_clean.clean(ga_ci,"campaign_impact"),
        ga_co          = googleapi_clean.clean(ga_co,"campaigns_outreach"),
        agg_bk         = agg_bk
    )