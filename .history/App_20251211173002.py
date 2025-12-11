# ====================================== #
# DASHBOARD ENERGI DUNIA - STREAMLIT     #
# ====================================== #

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 1️⃣ Load dan Persiapkan Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/owid-energy-data.csv")
    cols = ['country', 'year', 'primary_energy_consumption',
            'renewables_share_energy', 'carbon_intensity_elec',
            'gdp', 'population']
    df = df[cols].dropna()
    df['year'] = df['year'].astype(int)
    
    negara_fokus = ['China', 'United States', 'India', 'Indonesia', 'Brazil']
    df = df[df['country'].isin(negara_fokus)]
    
    scaler = MinMaxScaler()
    num_cols = ['primary_energy_consumption', 'renewables_share_energy', 
                'carbon_intensity_elec', 'gdp', 'population']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

df = load_data()

# -------------------------------
# 2️⃣ Sidebar Navigasi
# -------------------------------
st.sidebar.title("🌍 Analisis Energi Dunia")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Analisis Data", "📈 Visualisasi", "🤖 Prediksi Energi"]
)

# -------------------------------
# 3️⃣ Halaman: Analisis Data
# -------------------------------
if menu == "📊 Analisis Data":
    st.title("📊 Analisis Deskriptif & Diagnostik")

    # Dropdown negara
    negara = st.selectbox("Pilih Negara:", sorted(df['country'].unique()))

    data_negara = df[df['country'] == negara]

    st.subheader(f"Data Awal – {negara}")
    st.dataframe(data_negara.head())

    st.subheader(f"Ringkasan Statistik – {negara}")
    st.write(data_negara.describe())

    st.subheader(f"Korelasi GDP, Energi, dan Intensitas Karbon – {negara}")
    corr = data_negara[['gdp', 'primary_energy_consumption', 'carbon_intensity_elec']].corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    

# -------------------------------
# 4️⃣ Halaman: Visualisasi
# -------------------------------
elif menu == "📈 Visualisasi":
    st.title("📈 Visualisasi Energi")
    negara = st.selectbox("Pilih Negara:", df['country'].unique())
    
    data_negara = df[df['country'] == negara]
    
    st.subheader(f"Tren Konsumsi Energi - {negara}")
    fig, ax = plt.subplots()
    ax.plot(data_negara['year'], data_negara['primary_energy_consumption'], marker='o')
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Konsumsi Energi (Normalized)")
    st.pyplot(fig)
    
    st.subheader("Perbandingan Energi Terbarukan vs Intensitas Karbon")
    fig, ax = plt.subplots()
    sns.barplot(x='year', y='renewables_share_energy', data=data_negara, color='green', label='Energi Terbarukan')
    sns.lineplot(x='year', y='carbon_intensity_elec', data=data_negara, color='red', label='Intensitas Karbon', ax=ax)
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 5️⃣ Halaman: Prediksi
# -------------------------------
elif menu == "🤖 Prediksi Energi":
    st.title("🤖 Prediksi Konsumsi Energi (Random Forest)")

    X = df[['gdp', 'population', 'renewables_share_energy', 'carbon_intensity_elec']]
    y = df['primary_energy_consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    st.write(f"**Mean Squared Error (MSE): {mse:.4f}**")

    # Tambahkan arti variabel
    st.subheader("📘 Arti Setiap Variabel")
    st.markdown("""
    | Variabel             | Arti                       |
    | -------------------- | -------------------------- |
    | **GDP**              | Aktivitas ekonomi negara   |
    | **Population**       | Jumlah penduduk            |
    | **Renewables share** | Proporsi energi terbarukan |
    | **Carbon intensity** | Seberapa kotor listriknya  |
    """)

    st.subheader("Simulasi Prediksi Baru")
    gdp = st.slider("GDP (normalized)", 0.0, 1.0, 0.5)
    population = st.slider("Populasi (normalized)", 0.0, 1.0, 0.5)
    renew = st.slider("Renewables Share (normalized)", 0.0, 1.0, 0.5)
    carbon = st.slider("Carbon Intensity (normalized)", 0.0, 1.0, 0.5)

    new_pred = model.predict([[gdp, population, renew, carbon]])[0]
    st.success(f"🔮 Prediksi Konsumsi Energi (Normalized): **{new_pred:.4f}**")
