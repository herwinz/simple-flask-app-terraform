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

# ------------------- 1. AUTHENTICATION -------------------
TENANT_ID = "dd0188eb-d44b-486f-ba78-7e3af4140437"
CLIENT_ID = "1a2ac7e2-0986-457a-8cad-32a7cc813809"
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")  # Set dari environment
SUBSCRIPTION_ID = "c4cdee6c-a7eb-4983-86ca-7c13e37828e9"
RESOURCE_GROUP = "rg-sandbox"
APP_SERVICE_PLAN = "plan-mwvknzywowuzn"

token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/token"
token_data = {
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "resource": "https://management.azure.com/"
}
token = requests.post(token_url, data=token_data).json()["access_token"]

# ------------------- 2. AMBIL METRIK CPU & MEMORY -------------------
metrics = ["CpuPercentage", "MemoryPercentage"]
RESOURCE_ID = f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/{RESOURCE_GROUP}/providers/Microsoft.Web/serverfarms/{APP_SERVICE_PLAN}"

params = {
    "timespan": "P30D",  # 30 hari
    "interval": "PT1H",  # setiap jam
    "metricnames": ",".join(metrics),
    "aggregation": "Average"
}
headers = {"Authorization": f"Bearer {token}"}
url = f"https://management.azure.com{RESOURCE_ID}/providers/microsoft.insights/metrics?api-version=2024-02-01"

resp = requests.get(url, headers=headers, params=params).json()
if "value" not in resp:
    print("❌ Gagal ambil data metrik!")
    exit()

# ------------------- 3. PARSING METRIK -------------------
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

# ------------------- 4. PREPROCESS & FEATURE ENGINEERING -------------------
df_all = df_all.dropna()
df_all = df_all.tz_convert("Asia/Jakarta")
df_all.reset_index(inplace=True)
df_all["hour"] = df_all["timestamp"].dt.hour
df_all["day_of_week"] = df_all["timestamp"].dt.dayofweek

def classify_status(cpu, mem):
    if cpu < 30 and mem < 30:
        return 0  # underutilized
    elif cpu > 80 or mem > 80:
        return 2  # overutilized
    else:
        return 1  # optimal

df_all["status"] = df_all.apply(lambda row: classify_status(row["CpuPercentage"], row["MemoryPercentage"]), axis=1)

# ------------------- 5. TRAIN MODEL -------------------
X = df_all[["CpuPercentage", "MemoryPercentage", "hour", "day_of_week"]]
y = df_all["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print(f"📊 Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix Decision Tree
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt, labels=[0, 1, 2]), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"🌲 Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf, labels=[0, 1, 2]), annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------- 6. PREDIKSI CONTOH -------------------
def predict_scaling(cpu, mem, hour, day):
    input_df = pd.DataFrame([[cpu, mem, hour, day]], columns=["CpuPercentage", "MemoryPercentage", "hour", "day_of_week"])
    pred = rf_model.predict(input_df)[0]
    if pred == 0:
        return "⬇ Underutilized (Downscale App Plan)"
    elif pred == 1:
        return "✅ Optimal"
    else:
        return "⬆ Overutilized (Upscale App Plan)"

# Contoh Prediksi
example = predict_scaling(cpu=85, mem=90, hour=14, day=2)
print("📢 Scaling Recommendation:", example)

# ------------------- 7. PREDIKSI MASSAL KE SEMUA DATA -------------------
df_all["DecisionTree_Status"] = dt_model.predict(X)
df_all["RandomForest_Status"] = rf_model.predict(X)

# Ubah status angka ke label string
status_map = {
    0: "Underutilized",
    1: "Optimal",
    2: "Overutilized"
}
df_all["DecisionTree_Status"] = df_all["DecisionTree_Status"].map(status_map)
df_all["RandomForest_Status"] = df_all["RandomForest_Status"].map(status_map)

# Pastikan datetime tidak pakai timezone (biar bisa diexport ke Excel)
df_all["timestamp"] = df_all["timestamp"].dt.tz_localize(None)

# Simpan ke Excel
df_all.to_excel("azure_app_plan_metrics_with_predictions.xlsx", index=False)
print("📁 File disimpan dengan prediksi status ML: 'azure_app_plan_metrics_with_predictions.xlsx'")