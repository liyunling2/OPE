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
DATE_RANGE = DateRange(start_date="2025-01-01", end_date="today")

# gets list of all dimensions and metrics
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

# function to create datasets
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
        output_path = f"./data/marketing/googleAPI/{filename}.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")

        print(f"{dataset_name} exported: {len(df)} rows")

    except Exception as e:
        print(f"{dataset_name} FAILED: {e}")


# datasets building
    # BUSINESS PERFORMANCE
    # PAGE INTERACTION
    # ECOMMERCE FUNNEL
    # USER BEHAVIOR
    # GEO & DEMOGRAPHICS
# try:
#     key_dict = json.loads(JSON_TEXT.strip(), strict=False)
#     client = BetaAnalyticsDataClient.from_service_account_info(key_dict)

#     # BUSINESS PERFORMANCE DATASET
#     run_dataset(
#         client,
#         "business_performance",
#         dimensions=[
#             "date",
#             "defaultChannelGroup",
#             "country",
#             "deviceCategory",
#         ],
#         metrics=[
#             "totalUsers",
#             "sessions",
#             "engagementRate",
#             "purchaseRevenue",
#             # "transactions",
#             "averageRevenuePerUser",
#         ],
#         filename="business_performance.csv",
#     )

#     # PAGE INTERACTION DATASET
#     run_dataset(
#         client,
#         "page_interaction",
#         dimensions=[
#             "fullPageUrl",
#             "unifiedPagePathScreen",
#             "unifiedPageScreen",
#             "unifiedScreenClass",
#             "unifiedScreenName"
#         ],
#         metrics=[
#             "screenPageViews",
#             "screenPageViewsPerSession",
#             "screenPageViewsPerUser",
#             "engagedSessions",
#             "userEngagementDuration"
#         ],
#         filename="page_interaction.csv",
#     )

#     # ECOMMERCE FUNNEL DATASET
#     run_dataset(
#         client,
#         "ecommerce_funnel",
#         dimensions=[
#             "date",
#             "itemCategory",
#             "itemName",
#             "deviceCategory",
#         ],
#         metrics=[
#             "itemsViewed",
#             "itemsAddedToCart",
#             "itemRevenue",
#         ],
#         filename="ecommerce_funnel.csv",
#     )

#     # USER BEHAVIOR DATASET
#     run_dataset(
#         client,
#         "user_behavior",
#         dimensions=[
#             "date",
#             "userAgeBracket",
#             "userGender",
#             "deviceCategory",
#             "brandingInterest",
#         ],
#         metrics=[
#             "eventsPerSession",
#             "averageSessionDuration",
#             "engagedSessions",
#             "engagementRate",
#             "bounceRate",
#         ],
#         filename="user_behavior.csv",
#     )

#     # GEO & DEMOGRAPHICS DATASET
#     run_dataset(
#         client,
#         "geo_demographics",
#         dimensions=[
#             "date",
#             "country",
#             "city"
#         ],
#         metrics=[
#             "totalUsers",
#             "sessions",
#             "purchaseRevenue",
#             "engagementRate",
#         ],
#         filename="geo_demographics.csv",
#     )
#     print("=" * 40)
#     print("\nextracted all datasets")

# except Exception as e:
#     print("\n" + "=" * 40)
#     print(f"SCRIPT FAILED: {e}")
#     print("=" * 40)