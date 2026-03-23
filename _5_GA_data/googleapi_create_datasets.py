import csv
import json
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Metric,
    Dimension
)
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
# 2. CONFIGURATION
PROPERTY_ID = os.getenv("PROPERTY_ID")
JSON_TEXT = os.getenv("JSON_TEXT")
DATE_RANGE = DateRange(start_date="2024-01-01", end_date="today")

####### gets list of all dimensions and metrics
def getMetadata():
    try:
        key_dict = json.loads(JSON_TEXT.strip(), strict=False)
        client = BetaAnalyticsDataClient.from_service_account_info(key_dict)

        metadata = client.get_metadata(
            name=f"properties/{PROPERTY_ID}/metadata"
        )

        # dimensions
        with open("./data/marketing/googleAPI/metadata/dimensions_metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "api_name",
                "custom",
                "category",
                "deprecated_name",
                "description"
            ])

            for dim in metadata.dimensions:
                writer.writerow([
                    dim.api_name,
                    dim.custom_definition,
                    dim.category,
                    ",".join(dim.deprecated_api_names),
                    dim.description
                ])
        # metrics
        with open("./data/marketing/googleAPI/metadata/metrics_metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "api_name",
                "custom",
                "category",
                "deprecated_name",
                "description"
            ])

            for metric in metadata.metrics:
                writer.writerow([
                    metric.api_name,
                    metric.custom_definition,
                    metric.category,
                    ",".join(metric.deprecated_api_names),
                    metric.description
                ])

        print("list of dim & metrics retrieved")

    except Exception as e:
        print("\n" + "="*40)
        print(f"FAILED: {e}")
        print("="*40)

####### function to create datasets
def run_dataset(client, dataset_name, dimensions, metrics, filename):
    try:
        all_rows = []
        offset = 0
        page_size = 100000  

        while True:
            request = RunReportRequest(
                property=f"properties/{PROPERTY_ID}",
                dimensions=[Dimension(name=d) for d in dimensions],
                metrics=[Metric(name=m) for m in metrics],
                date_ranges=[DATE_RANGE],
                limit=page_size,
                offset=offset,
            )

            response = client.run_report(request)

            if not response.rows:
                break

            for row in response.rows:
                row_data = {}
                for i, dim in enumerate(dimensions):
                    row_data[dim] = row.dimension_values[i].value
                for i, met in enumerate(metrics):
                    row_data[met] = row.metric_values[i].value
                all_rows.append(row_data)

            offset += page_size

        df = pd.DataFrame(all_rows)
        output_path = f"../data/marketing/googleAPI/{filename}.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        
        # output_path = f"./data/marketing/googleAPI/{filename}.csv"
        # df.to_csv(output_path, index=False)

        print(f"{dataset_name} exported: {len(df)} rows")

    except Exception as e:
        print(f"{dataset_name} FAILED: {e}")


# no valuable data available - dimensions
    # adFormat,adSourceName,adUnitName, sessionCampaignId, sessionCampaignName, CampaignId, CampaignName, sessionManualMarketingTactic


## marketing strategy -> suggest platform to promote on, types of campaign/notifications -> increase user engagement & checkouts [could further recommend to target different user demographics]
####### datasets building ########
# session source
    # identify the platform of ads and campaign promotion that brings user to engage with HH
# campaign impact
    #  show restaurant's campaign items as ecommerce funnel - viewed, added to cart, checked out as revenue
# time series data; user engagement
    # identify engagement trends in different campaign's time period
# time series data; new users
# time series data; unique restaurants
# user demographics
    # possibly find changes in user demographics over time; draw links between campaign success to demographics

def createDatasets():   
    try:
        key_dict = json.loads(JSON_TEXT.strip(), strict=False)
        client = BetaAnalyticsDataClient.from_service_account_info(key_dict)
        print("=" * 40)

        # # session source 
        run_dataset(
            client,
            "session_source",
            dimensions=[
                "yearMonth",
                "sessionSource",
                "sessionDefaultChannelGroup"
            ],
            metrics=[
                "sessions",
                # QUALITY — what happened in those sessions
                'engagedSessions',      
                'bounceRate',           
                # OUTCOMES — what those sessions resulted in 
                'keyEvents:booking',          
                'keyEvents:begin_checkout',   
                'keyEvents:ecommerce_purchase',
                'purchaseRevenue',            
                'transactions'
            ],
            filename="session_source",
        )

        # campaigns_outreach recorded
        run_dataset(
            client,
            "campaigns_outreach",
            dimensions=[
                "yearMonth",
                "campaignId",
                "campaignName"
            ],
            metrics=[
                "sessions"
            ],
            filename="campaigns_outreach",
        )
        # campaign impact 
        run_dataset(
            client,
            "campaign_impact",
            dimensions=[
                "yearMonth",
                "itemId",
                "itemName",
            ],
            metrics=[
                "itemsViewed",
                "itemsAddedToCart",
                "itemsPurchased",
                "itemRevenue"
            ],
            filename="campaign_impact",
        )

        ### not allowed
        # run_dataset(
        #     client,
        #     "campaign_rest",
        #     dimensions=[
        #         "itemId",
        #         "itemName",
        #         "campaignId",
        #         "campaignName",
        #     ],
        #     metrics=[
        #     ],
        #     filename="campaign_rest",
        # )


    # time series
    # new users
        run_dataset(
            client,
            "agg_new_users",
            dimensions=[
                "yearMonth",
                "firstUserSourcePlatform",
            ],
            metrics=[
                "newUsers",
                "activeUsers"
            ],
            filename="agg_new_users",
        )
    # user_activity
        run_dataset(
            client,
            "agg_user_activity",
            dimensions=[
                "yearMonth",
            ],
            metrics=[
                "activeUsers",
                "engagedSessions",
                "sessionsPerUser",
                "userEngagementDuration",
                "addToCarts", #The number of times users added items to their shopping carts.
                "checkouts", #The number of times users started the checkout process. This metric counts the occurrence of the `begin_checkout` event.
                "ecommercePurchases", #The number of times users completed a purchase. This metric counts `purchase` events; this metric does not count `in_app_purchase` and subscription events.
                "totalPurchasers", #The number of users that logged purchase events for the time period selected.
                "purchaseRevenue", #The sum of revenue from purchases minus refunded transaction revenue made in your app or site. Purchase revenue sums the revenue for these events: `purchase`, `ecommerce_purchase`, `in_app_purchase`, `app_store_subscription_convert`, and `app_store_subscription_renew`. Purchase revenue is specified by the `value` parameter in tagging.
                "totalRevenue", #The sum of revenue from purchases, subscriptions, and advertising (Purchase revenue plus Subscription revenue plus Ad revenue) minus refunded transaction revenue.
            ],
            filename="agg_user_activity",
        )

    # unique restaurants
        run_dataset(
            client,
            "agg_restaurants",
            dimensions=[
                "year",
                "itemId",
                "itemName",
            ], 
            metrics=[
            ],
            filename="agg_restaurants",
        )
        
    # user demographic
        #### aggregated by month/year
        run_dataset(
            client,
            "user_demographics",
            dimensions=[
                "yearMonth",
                "userAgeBracket",
                "userGender"
            ],
            metrics=[
                "activeUsers",
                "engagedSessions",
                "ecommercePurchases",
                "engagementRate",
                "purchaseRevenue",
            ],
            filename="user_demographics",
        )
        
    # # PAGE INTERACTION 
        # run_dataset(
        #     client,
        #     "page_interaction",
        #     dimensions=[
        #         "fullPageUrl",
        #         "unifiedPagePathScreen",
        #         "unifiedPageScreen",
        #         "unifiedScreenClass",
        #         "unifiedScreenName"
        #     ],
        #     metrics=[
        #         "screenPageViews",
        #         "screenPageViewsPerSession",
        #         "screenPageViewsPerUser",
        #         "engagedSessions",
        #         "userEngagementDuration"
        #     ],
        #     filename="page_interaction",
        # )
        print("=" * 40)
        print("\nextracted all datasets")

    except Exception as e:
        print("\n" + "=" * 40)
        print(f"SCRIPT FAILED: {e}")
        print("=" * 40)

if __name__ == "__main__":

    ## create dim and metrics metadata
    # getMetadata()

    ## create datasets
    createDatasets()