import requests
import json
import os
import pandas as pd
from datetime import datetime

# ------------------- SETUP AUTHENTICATION -------------------
TENANT_ID = "af2c0734-cb42-464f-b6bf-2a241b6ada56"
CLIENT_ID = "aa6bdeca-20bc-4241-8772-a7df362b8a39"
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")  # Set dari environment
SUBSCRIPTION_ID = "3bff15a8-79cf-44d3-b98f-94606c8f3a60"

# OAuth
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/token"
data = {
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "resource": "https://management.azure.com/"
}
response = requests.post(TOKEN_URL, data=data)
token_data = response.json()
if "access_token" not in token_data:
    print("❌ Gagal dapat token")
    print(json.dumps(token_data, indent=4))
    exit()
token = token_data["access_token"]
print("✅ Token didapatkan")

# ------------------- KONFIGURASI APP SERVICE PLAN -------------------
RESOURCE_GROUP_NAME = "rg-dev"
APP_SERVICE_PLAN_NAME = "plan-yzcznwy1y2zhm"  # Sesuaikan dengan milikmu

RESOURCE_ID = f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/{RESOURCE_GROUP_NAME}/providers/Microsoft.Web/serverfarms/{APP_SERVICE_PLAN_NAME}"
METRICS_URL = f"https://management.azure.com{RESOURCE_ID}/providers/microsoft.insights/metrics?api-version=2024-02-01"

# App Service Plan Metrics
METRIC_NAMES = [
    "CpuPercentage",
    "MemoryPercentage",
    "DiskQueueLength",
    "HttpQueueLength",
    "BytesReceived",
    "BytesSent"
]

params = {
    "timespan": "P5D",
    "interval": "PT5M",
    "metricnames": ",".join(METRIC_NAMES),
    "aggregation": "Average"
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.get(METRICS_URL, headers=headers, params=params)
data = response.json()

if response.status_code != 200 or "value" not in data:
    print(f"❌ Gagal ambil metrik. Status Code: {response.status_code}")
    print(json.dumps(data, indent=4))
    exit()

print("✅ Data App Service Plan berhasil diambil!")

# ------------------- PARSING KE DATAFRAME -------------------
metrics_data = {}

for metric in data["value"]:
    metric_name = metric["name"]["value"]
    time_series = []
    for timeseries in metric["timeseries"]:
        for point in timeseries["data"]:
            if "average" in point and "timeStamp" in point:
                time_series.append({
                    "Time": point["timeStamp"],
                    metric_name: point["average"]
                })
    if time_series:
        metrics_data[metric_name] = pd.DataFrame(time_series)

df = None
for key, df_metric in metrics_data.items():
    df_metric["Time"] = pd.to_datetime(df_metric["Time"])
    df_metric.set_index("Time", inplace=True)
    if df is None:
        df = df_metric
    else:
        df = df.join(df_metric, how="outer")

# Timezone & Formatting
df.index = df.index.tz_convert("Asia/Jakarta")
df.reset_index(inplace=True)
df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Pilih kolom
selected_columns = ["Time"] + METRIC_NAMES
df = df[selected_columns]

# Simpan sebagai file
df.to_csv("app_service_plan_metrics.csv", index=False, float_format="%.2f")
df.to_excel("app_service_plan_metrics.xlsx", index=False, float_format="%.2f")

print("✅ Data metrics App Service Plan disimpan ke CSV dan Excel.")
