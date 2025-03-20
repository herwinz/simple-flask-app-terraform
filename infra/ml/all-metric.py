import requests
import json
import os
import pandas as pd
from datetime import datetime
from openpyxl import Workbook

# ------------------- SETUP AUTHENTICATION -------------------
TENANT_ID = "af2c0734-cb42-464f-b6bf-2a241b6ada56"
CLIENT_ID = "aa6bdeca-20bc-4241-8772-a7df362b8a39"
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")  # Pastikan variabel ini sudah diatur
SUBSCRIPTION_ID = "3bff15a8-79cf-44d3-b98f-94606c8f3a60"

# Azure OAuth URL
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/token"

# Request access token
data = {
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "resource": "https://management.azure.com/"
}

response = requests.post(TOKEN_URL, data=data)
token_data = response.json()

# Periksa apakah token berhasil didapatkan
if "access_token" not in token_data:
    print("❌ ERROR: Gagal mendapatkan token. Periksa kredensial autentikasi.")
    print(json.dumps(token_data, indent=4))
    exit()

token = token_data["access_token"]
print("✅ Access Token didapatkan!")

# ------------------- AMBIL DATA METRICS -------------------
RESOURCE_ID = f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/rg-dev/providers/Microsoft.Web/sites/app-web-yzcznwy1y2zhm"

METRICS_URL = f"https://management.azure.com{RESOURCE_ID}/providers/microsoft.insights/metrics?api-version=2024-02-01"

# Daftar metrik yang valid berdasarkan error sebelumnya
METRIC_NAMES = [
    "CpuTime",
    "MemoryWorkingSet",
    "Requests",
    "Http5xx",
    "Http4xx",
    "AverageResponseTime",  # ✅ Mengganti ResponseTime yang tidak valid
    "InstanceCount",  # Menambahkan jumlah instance
]

params = {
    "timespan": "P5D",  # Ambil data 1 hari terakhir
    "interval": "PT5M",  # Data setiap 5 menit
    "metricnames": ",".join(METRIC_NAMES),
    "aggregation": "Average"
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.get(METRICS_URL, headers=headers, params=params)
data = response.json()

# Debug: Periksa status response dan tampilkan isi JSON
if response.status_code != 200 or "value" not in data:
    print(f"❌ ERROR: Gagal mengambil data metrik. Status Code: {response.status_code}")
    print(json.dumps(data, indent=4))
    exit()

print("✅ Data berhasil diambil dari Azure Monitor API!")

# ------------------- PARSING DATA KE PANDAS DATAFRAME -------------------
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

    # Simpan ke dictionary untuk merging nanti
    if time_series:
        metrics_data[metric_name] = pd.DataFrame(time_series)

# Gabungkan semua metrik berdasarkan Time (Inner Join)
df = None
for key, df_metric in metrics_data.items():
    df_metric["Time"] = pd.to_datetime(df_metric["Time"])
    df_metric.set_index("Time", inplace=True)

    if df is None:
        df = df_metric
    else:
        df = df.join(df_metric, how="outer")

# 🔹 Konversi Time ke WIB (UTC+7)
df.index = df.index.tz_convert("Asia/Jakarta")

# 🔹 Ubah Format Time tanpa Timezone
df.reset_index(inplace=True)
df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")  # Format lengkap dengan tanggal

# 🔹 Konversi CPU Time dari Detik ke Menit
if "CpuTime" in df:
    df["CpuTime (minutes)"] = df["CpuTime"] / 60  

# 🔹 Konversi Memory dari Bytes ke Megabytes (MB)
if "MemoryWorkingSet" in df:
    df["Memory (MB)"] = df["MemoryWorkingSet"] / (1024 * 1024)

# 🔹 Definisikan Total Memory berdasarkan App Service Plan B1 (1.75 GB = 1792 MB)
TOTAL_MEMORY_MB = 1792  
if "Memory (MB)" in df:
    df["Memory_Percentage"] = (df["Memory (MB)"] / TOTAL_MEMORY_MB) * 100

# 🔹 Hitung CPU Percentage
INTERVAL_SECONDS = 5 * 60  # 5 menit dalam detik
CPU_CORES = 1  # Jumlah CPU Core

if "CpuTime" in df:
    df["CPU_Percentage"] = (df["CpuTime"] / (CPU_CORES * INTERVAL_SECONDS)) * 100

# 🔹 Pilih Kolom yang Dibutuhkan
selected_columns = ["Time"]
if "CpuTime (minutes)" in df:
    selected_columns.append("CpuTime (minutes)")
if "Memory (MB)" in df:
    selected_columns.append("Memory (MB)")
if "Memory_Percentage" in df:
    selected_columns.append("Memory_Percentage")
if "CPU_Percentage" in df:
    selected_columns.append("CPU_Percentage")
if "Requests" in df:
    selected_columns.append("Requests")
if "Http5xx" in df:
    selected_columns.append("Http5xx")
if "Http4xx" in df:
    selected_columns.append("Http4xx")
if "AverageResponseTime" in df:
    selected_columns.append("AverageResponseTime")  # ✅ Menggunakan AverageResponseTime yang valid
if "InstanceCount" in df:
    selected_columns.append("InstanceCount")

df = df[selected_columns]

# Simpan dataset sebagai CSV dan Excel
df.to_csv("all_azure_metrics.csv", index=False, float_format="%.2f")

# Simpan sebagai Excel
excel_filename = "azure_metrics.xlsx"
df.to_excel(excel_filename, index=False, float_format="%.2f")

print(f"✅ Data berhasil dikonversi ke format yang lebih mudah dibaca! ({excel_filename})")
