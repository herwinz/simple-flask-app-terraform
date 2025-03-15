import requests
import json
import os
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from prophet import Prophet

# ------------------- SETUP AUTHENTICATION -------------------
# Ganti dengan informasi dari App Registration di Azure
TENANT_ID = "af2c0734-cb42-464f-b6bf-2a241b6ada56"
CLIENT_ID = "aa6bdeca-20bc-4241-8772-a7df362b8a39"
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")  # Menggunakan variabel lingkungan untuk keamanan
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
token = response.json()["access_token"]
print("✅ Access Token didapatkan!")

# ------------------- AMBIL DATA METRICS -------------------
# Ganti dengan Workspace ID dan Resource ID dari App Service
RESOURCE_ID = f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/rg-dev/providers/Microsoft.Web/sites/app-web-yzcznwy1y2zhm"

# Azure Monitor API Endpoint
METRICS_URL = f"https://management.azure.com{RESOURCE_ID}/providers/microsoft.insights/metrics?api-version=2024-02-01"

# Parameter API Request
params = {
    # "timespan": "P90D",  # Ambil data 90 hari terakhir
    # "timespan": "P7D",  # Ambil data 7 hari terakhir
    # "timespan": "P2D",  # Ambil data 2 hari terakhir
    "timespan": "P1D",  # Ambil data 1 hari terakhir
    "interval": "PT5M",  # Data setiap 5 menit
    "metricnames": "CpuTime,MemoryWorkingSet",  # ✅ Metric yang valid
    "aggregation": "Average"
}

# Headers dengan Token Autentikasi
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Request Data dari Azure Monitor API
response = requests.get(METRICS_URL, headers=headers, params=params)
data = response.json()

# Simpan hasil JSON ke file
with open("azure_metrics.json", "w") as f:
    json.dump(data, f, indent=4)

print("✅ Data berhasil diambil dari Azure Monitor API!")

# ------------------- PARSING DATA KE PANDAS DATAFRAME -------------------
# Load JSON yang baru diunduh
with open("azure_metrics.json", "r") as f:
    data = json.load(f)

# Ambil data CPU Time & Memory
cpu_values = []
memory_values = []

for metric in data["value"]:
    name = metric["name"]["value"]
    for timeseries in metric["timeseries"]:
        for data_point in timeseries["data"]:
            if "average" in data_point and "timeStamp" in data_point:
                if name == "CpuTime":
                    cpu_values.append({"Time": data_point["timeStamp"], "CpuTime": data_point["average"]})
                elif name == "MemoryWorkingSet":
                    memory_values.append({"Time": data_point["timeStamp"], "Memory": data_point["average"]})

# Buat DataFrame untuk CPU Time dan Memory
df_cpu = pd.DataFrame(cpu_values)
df_memory = pd.DataFrame(memory_values)

# Gabungkan data berdasarkan timestamp
df = pd.merge(df_cpu, df_memory, on="Time")

# 🔹 Konversi Time ke WIB (UTC+7)
df["Time"] = pd.to_datetime(df["Time"]).dt.tz_convert("Asia/Jakarta")

# 🔹 Ubah Format Time tanpa Timezone
df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")  # Format lengkap dengan tanggal

# 🔹 Konversi CPU Time dari Detik ke Menit
df["CpuTime (minutes)"] = df["CpuTime"] / (60 * 60)  # Dari detik ke menit yang benar

# 🔹 Konversi Memory dari Bytes ke Megabytes (MB)
df["Memory (MB)"] = df["Memory"] / (1024 * 1024)

# 🔹 Hitung CPU Percentage
df["CPU_Percentage"] = (df["CpuTime"] / df["CpuTime"].max()) * 100


# 🔹 Pilih Kolom yang Dibutuhkan
df = df[["Time", "CpuTime (minutes)", "Memory (MB)", "CPU_Percentage"]]

# Simpan dataset sebagai CSV yang lebih mudah dibaca
df.to_csv("azure_metrics.csv", index=False, float_format="%.2f")

print("✅ Data berhasil dikonversi ke format yang lebih mudah dibaca!")

# ------------------- ANALISIS DATA DENGAN PROPHET -------------------
# Siapkan data untuk Prophet
df_prophet = df.rename(columns={"Time": "ds", "CPU_Percentage": "y"})
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])  
df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)  # Hapus timezone

# Buat model Prophet
model = Prophet()
model.fit(df_prophet)

# Prediksi 30 hari ke depan
future = model.make_future_dataframe(periods=30, freq="D")
forecast = model.predict(future)

# Plot hasil prediksi Prophet
model.plot(forecast)
plt.title("Prediksi Penggunaan CPU dengan Prophet")
plt.show()

# ------------------- ANALISIS DATA DENGAN ARIMA -------------------
# Gunakan CPU Usage sebagai time series
cpu_series = df["CPU_Percentage"]

# Buat Model ARIMA (p=5, d=1, q=2)
model_arima = ARIMA(cpu_series, order=(5,1,2))
model_fit = model_arima.fit()

# Prediksi 30 hari ke depan
forecast_arima = model_fit.forecast(steps=30)

# Plot hasil prediksi ARIMA
plt.figure(figsize=(10,5))
plt.plot(cpu_series, label="Data Aktual")
plt.plot(pd.date_range(cpu_series.index[-1], periods=30, freq="D"), forecast_arima, label="Prediksi ARIMA", color="red")
plt.legend()
plt.title("Prediksi Penggunaan CPU dengan ARIMA")
plt.show()

print("✅ Analisis dengan Prophet & ARIMA selesai!")