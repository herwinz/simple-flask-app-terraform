import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# ------------------- 1. AUTHENTICATION -------------------
TENANT_ID = "af2c0734-cb42-464f-b6bf-2a241b6ada56"
CLIENT_ID = "aa6bdeca-20bc-4241-8772-a7df362b8a39"
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
SUBSCRIPTION_ID = "3bff15a8-79cf-44d3-b98f-94606c8f3a60"
RESOURCE_GROUP = "rg-dev"
APP_SERVICE_PLAN = "plan-yzcznwy1y2zhm"

# Get Azure token
token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/token"
token_data = {
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "resource": "https://management.azure.com/"
}
token = requests.post(token_url, data=token_data).json()["access_token"]

# ------------------- 2. AMBIL METRIK LENGKAP DARI APP SERVICE PLAN -------------------
metrics = ['CpuPercentage', 'MemoryPercentage', 'DiskQueueLength', 'HttpQueueLength',
           'BytesReceived', 'BytesSent']  # InstanceCount dihapus karena tidak valid
RESOURCE_ID = f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/{RESOURCE_GROUP}/providers/Microsoft.Web/serverfarms/{APP_SERVICE_PLAN}"

params = {
    "timespan": "P30D",
    "interval": "PT5M",  # PT1M,PT5M,PT15M,PT30M,PT1H,PT6H,PT12H,P1D
    "metricnames": ",".join(metrics),
    "aggregation": "Average"
}
headers = {"Authorization": f"Bearer {token}"}
url = f"https://management.azure.com{RESOURCE_ID}/providers/microsoft.insights/metrics?api-version=2024-02-01"

resp = requests.get(url, headers=headers, params=params).json()
if "value" not in resp:
    print("\u274c Gagal ambil data metrik!")
    print(json.dumps(resp, indent=2))  # Debug response
    exit()

# ------------------- 3. PARSING SEMUA METRIK -------------------
df_all = None
for metric in resp["value"]:
    name = metric["name"]["value"]
    points = []
    for ts in metric["timeseries"]:
        for item in ts["data"]:
            if "average" in item:
                points.append({
                    "timestamp": pd.to_datetime(item["timeStamp"]),
                    name: item["average"]
                })
    df_metric = pd.DataFrame(points).set_index("timestamp")
    df_all = df_metric if df_all is None else df_all.join(df_metric, how="outer")

# ------------------- 4. FEATURE ENGINEERING LANJUTAN -------------------
df_all = df_all.dropna()
df_all = df_all.tz_convert("Asia/Jakarta")
df_all.reset_index(inplace=True)
df_all["hour"] = df_all["timestamp"].dt.hour
df_all["day_of_week"] = df_all["timestamp"].dt.dayofweek
df_all["is_weekend"] = df_all["day_of_week"].isin([5, 6]).astype(int)
df_all["cpu_trend"] = df_all["CpuPercentage"].diff().fillna(0)
df_all["mem_trend"] = df_all["MemoryPercentage"].diff().fillna(0)

# Label Status
def classify_status(cpu, mem):
    if cpu < 30 and mem < 30:
        return 0
    elif cpu > 80 or mem > 80:
        return 2
    else:
        return 1

df_all["status"] = df_all.apply(lambda row: classify_status(row["CpuPercentage"], row["MemoryPercentage"]), axis=1)

# ------------------- 5. MODEL TRAINING -------------------
features = [
    "CpuPercentage", "MemoryPercentage", "DiskQueueLength", "HttpQueueLength",
    "BytesReceived", "BytesSent",
    "hour", "day_of_week", "is_weekend", "cpu_trend", "mem_trend"
]
X = df_all[features]
y = df_all["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ------------------- 6. EVALUASI MODEL -------------------
print(f"\U0001F4CA Decision Tree Accuracy: {{accuracy_score(y_test, y_pred_dt):.2f}}")
print(classification_report(y_test, y_pred_dt))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt, labels=[0, 1, 2]), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(f"\U0001F333 Random Forest Accuracy: {{accuracy_score(y_test, y_pred_rf):.2f}}")
print(classification_report(y_test, y_pred_rf))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf, labels=[0, 1, 2]), annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------- 7. SIMPAN PREDIKSI -------------------
df_all["DecisionTree_Status"] = dt_model.predict(X)
df_all["RandomForest_Status"] = rf_model.predict(X)

status_map = { 0: "Underutilized", 1: "Optimal", 2: "Overutilized" }
df_all["DecisionTree_Status"] = df_all["DecisionTree_Status"].map(status_map)
df_all["RandomForest_Status"] = df_all["RandomForest_Status"].map(status_map)

# Simpan ke Excel
df_all["timestamp"] = df_all["timestamp"].dt.tz_localize(None)
df_all.to_excel("azure_app_plan_metrics_with_predictions_enhanced.xlsx", index=False)
print("\U0001F4C1 File disimpan: azure_app_plan_metrics_with_predictions_enhanced.xlsx")