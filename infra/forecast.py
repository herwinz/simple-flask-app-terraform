import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Membaca data dari file Excel
file_path = "Azure_App_Service_Data_Adjusted.xlsx"

# Membaca data historis
historical_data = pd.read_excel(file_path, sheet_name="DATA_A")
print("Data Historis:")
print(historical_data.head())

# Membaca data referensi Azure App Service Plans
azure_tiers = pd.read_excel(file_path, sheet_name="Azure Tiers")
print("Data Azure App Service Plans:")
print(azure_tiers.head())

# Pastikan kolom CPU dan Memory adalah angka
azure_tiers['CPU'] = pd.to_numeric(azure_tiers['CPU'], errors='coerce').fillna(0)
azure_tiers['Memory (GB)'] = pd.to_numeric(azure_tiers['Memory (GB)'], errors='coerce').fillna(0)

# Menyiapkan data untuk Prophet
prophet_data = historical_data.rename(columns={'datetime': 'ds', 'cpu_usage': 'y'})

# Inisialisasi dan pelatihan model Prophet untuk CPU usage
model_cpu = Prophet()
model_cpu.fit(prophet_data)

# Membuat prediksi untuk 30 hari ke depan
future_cpu = model_cpu.make_future_dataframe(periods=30)
forecast_cpu = model_cpu.predict(future_cpu)

# Tambahkan kolom memory_usage dan request_count rata-rata
forecast_cpu['memory_usage'] = historical_data['memory_usage'].mean()
forecast_cpu['request_count'] = historical_data['request_count'].mean()

# Fungsi untuk menentukan tier Azure berdasarkan CPU dan Memory
def map_to_tier(row, tiers):
    for _, tier in tiers.iterrows():
        if row['yhat'] <= tier['CPU'] and row['memory_usage'] <= tier['Memory (GB)']:
            return tier['Name']
    return "Upgrade Needed"

# Menentukan tier untuk setiap prediksi
forecast_cpu['tier'] = forecast_cpu.apply(lambda row: map_to_tier(row, azure_tiers), axis=1)

# Menampilkan hasil prediksi dan tier yang direkomendasikan
forecast_result = forecast_cpu[['ds', 'yhat', 'memory_usage', 'request_count', 'tier']].tail(30)

# Visualisasi hasil prediksi CPU
fig = model_cpu.plot(forecast_cpu)
plt.title("CPU Usage Prediction")
plt.xlabel("Date")
plt.ylabel("CPU Usage (%)")
plt.show()

# Menampilkan hasil akhir
print("Hasil Prediksi dan Tier yang Direkomendasikan:")
print(forecast_result)

# Mengambil hasil rekomendasi tier terakhir
recommended_tier = forecast_result['tier'].iloc[-1]

# Menentukan SKU yang sesuai berdasarkan tier yang direkomendasikan
if recommended_tier == "Basic B2":
    sku_name = "B2"
elif recommended_tier == "Basic B1":
    sku_name = "B1"
else:
    sku_name = "B1"  # Default ke "B1" jika tidak ada yang cocok

# Membaca dan mengubah file main.tf
terraform_file_path = "main.tf"

try:
    with open(terraform_file_path, 'r') as file:
        terraform_file = file.read()

    # Mengganti sku_name sesuai dengan hasil rekomendasi
    terraform_file = terraform_file.replace('sku_name = "B1"', f'sku_name = "{sku_name}"')

    # Menyimpan perubahan ke file main.tf
    with open(terraform_file_path, 'w') as file:
        file.write(terraform_file)

    print(f"SKU name dalam main.tf telah diubah menjadi {sku_name}")

except Exception as e:
    print(f"Terjadi kesalahan saat mengubah file: {e}")
