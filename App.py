# ====================================== #
# DASHBOARD ENERGI DUNIA - STREAMLIT     #
# ====================================== #

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="World Energy Dashboard", layout="wide", page_icon="⚡")

# --- 1️⃣ LOAD DATA (CACHED) ---
@st.cache_data
def load_data():
    # Pastikan file CSV tersedia di direktori yang benar
    df = pd.read_csv("data/owid-energy-data.csv")
    cols = [
        'country', 'year', 'primary_energy_consumption',
        'renewables_share_energy', 'carbon_intensity_elec',
        'gdp', 'population'
    ]
    df = df[cols].dropna()
    df['year'] = df['year'].astype(int)
    # Fokus pada 5 negara besar
    negara_fokus = ['China', 'United States', 'India', 'Indonesia', 'Brazil']
    return df[df['country'].isin(negara_fokus)]

df = load_data()

# Data untuk visualisasi (Konversi Unit)
df_display = df.copy()
df_display['gdp_trillion'] = df_display['gdp'] / 1e12
df_display['population_million'] = df_display['population'] / 1e6

# --- 2️⃣ MODEL TRAINING (CACHED RESOURCE) ---
@st.cache_resource
def train_energy_model(data_negara):
    X = data_negara[['gdp','population','renewables_share_energy','carbon_intensity_elec']]
    y = data_negara['primary_energy_consumption']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Hitung metrik evaluasi
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, scaler, metrics, X_test_scaled, y_test

# --- 3️⃣ SIDEBAR ---
st.sidebar.title("🌍 Navigasi Analisis")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Analisis Data", "📈 Visualisasi", "🤖 Prediksi Energi", "🌍 Peta Energi Dunia"]
)

# ============================================================
# 4️⃣ HALAMAN: ANALISIS DATA
# ============================================================
if menu == "📊 Analisis Data":
    st.title("📊 Analisis Deskriptif & Korelasi")
    mode = st.radio("Pilih Mode:", ["Analisis Satu Negara", "Perbandingan Dua Negara"], horizontal=True)

    if mode == "Analisis Satu Negara":
        negara = st.selectbox("Pilih Negara:", sorted(df['country'].unique()))
        data = df_display[df_display['country'] == negara]

        st.subheader(f"Ringkasan Statistik – {negara}")
        st.dataframe(data[['year','gdp_trillion','population_million','primary_energy_consumption','renewables_share_energy']].tail())

        st.subheader("Matriks Korelasi")
        corr = data[['gdp','primary_energy_consumption','carbon_intensity_elec']].corr()
        
        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="RdYlGn", ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📝 Interpreter Korelasi")
            c_gdp = corr.loc['gdp', 'primary_energy_consumption']
            c_carb = corr.loc['primary_energy_consumption', 'carbon_intensity_elec']
            
            msg = f"Di **{negara}**, hubungan antara GDP dan energi adalah **{c_gdp:.2f}**. "
            if abs(c_gdp) > 0.8:
                msg += "Ini menunjukkan ekonomi sangat bergantung pada konsumsi energi."
            else:
                msg += "Hubungan ini moderat, mengindikasikan adanya faktor lain yang memengaruhi ekonomi."
                
            st.info(msg)
            st.write(f"Korelasi Energi-Karbon: **{c_carb:.2f}**")
            if c_carb < 0:
                st.success("Bagus! Kenaikan energi tidak dibarengi kenaikan emisi (Decoupling).")

    else:
        c1, c2 = st.columns(2)
        with c1: n1 = st.selectbox("Negara 1:", df['country'].unique(), index=0) 
        with c2: n2 = st.selectbox("Negara 2:", df['country'].unique(), index=1)
        
        m1 = df[df['country'] == n1].mean(numeric_only=True)
        m2 = df[df['country'] == n2].mean(numeric_only=True)
        
        st.table(pd.DataFrame({n1: m1, n2: m2}).T[['gdp', 'primary_energy_consumption', 'renewables_share_energy']])
        
        st.markdown("### 📝 Interpreter Perbandingan")
        gap = abs(m1['renewables_share_energy'] - m2['renewables_share_energy'])
        winner = n1 if m1['renewables_share_energy'] > m2['renewables_share_energy'] else n2
        st.success(f"**{winner}** memimpin transisi energi dengan selisih **{gap:.2f}%** lebih tinggi dalam porsi energi terbarukan.")

# ============================================================
# 5️⃣ HALAMAN: VISUALISASI
# ============================================================
elif menu == "📈 Visualisasi":
    st.title("📈 Visualisasi Tren Historis")
    negara = st.selectbox("Pilih Negara:", df['country'].unique())
    data = df_display[df_display['country'] == negara].sort_values("year")

    tab1, tab2 = st.tabs(["🚀 Konsumsi & GDP", "🍃 Transisi Hijau"])

    with tab1:
        st.plotly_chart(px.line(data, x='year', y=['primary_energy_consumption', 'gdp_trillion'], title="Pertumbuhan Ekonomi vs Energi"), use_container_width=True)
        # Interpreter Tren
        diff = ((data['primary_energy_consumption'].iloc[-1] / data['primary_energy_consumption'].iloc[0]) - 1) * 100
        st.markdown(f"**Analisis Tren:** Konsumsi energi meningkat sebesar **{diff:.1f}%** sejak awal periode data.")

    with tab2:
        st.plotly_chart(px.area(data, x='year', y='renewables_share_energy', color_discrete_sequence=['green'], title="Porsi Energi Terbarukan"), use_container_width=True)
        
        # Green Interpreter
        last_val = data['renewables_share_energy'].iloc[-1]
        st.markdown("### 📝 Interpreter Transisi Hijau")
        if last_val > 20:
            st.success(f"Status: **Pionir Hijau**. Dengan porsi {last_val:.1f}%, {negara} berada di jalur yang benar menuju Net Zero.")
        else:
            st.warning(f"Status: **Ketergantungan Fosil**. Porsi {last_val:.1f}% masih cukup rendah untuk skala ekonomi besar.")

# ============================================================
# 6️⃣ HALAMAN: PREDIKSI ENERGI
# ============================================================
elif menu == "🤖 Prediksi Energi":
    st.title("🤖 Prediksi Konsumsi (Random Forest)")
    negara = st.selectbox("Pilih Negara:", df['country'].unique())
    df_neg = df[df['country'] == negara]

    model, scaler, metrics, X_test_scaled, y_test = train_energy_model(df_neg)
    
    # Metrik Evaluasi
    st.subheader("📊 Performa Model (Error Metrics)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score (Akurasi)", f"{metrics['r2']:.4f}")
    m2.metric("MAE", f"{metrics['mae']:.2f}")
    m3.metric("MSE", f"{metrics['mse']:.2f}")

    with st.expander("📝 Apa arti angka ini?"):
        st.write(f"Model ini memiliki akurasi $R^2$ sebesar **{metrics['r2']*100:.1f}%**. Artinya, variabel input (GDP, Populasi, dll) mampu menjelaskan sebagian besar pola energi di {negara}.")

    st.markdown("---")
    st.subheader("🎛 Simulasi Input")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        g_in = st.slider("GDP (Triliun)", float(df_neg['gdp'].min()/1e12), float(df_neg['gdp'].max()/1e12), float(df_neg['gdp'].mean()/1e12))
        p_in = st.slider("Populasi (Juta)", float(df_neg['population'].min()/1e6), float(df_neg['population'].max()/1e6), float(df_neg['population'].mean()/1e6))
    with col_in2:
        r_in = st.slider("Renewable (%)", 0.0, 100.0, float(df_neg['renewables_share_energy'].mean()))
        c_in = st.slider("Carbon Intensity", float(df_neg['carbon_intensity_elec'].min()), float(df_neg['carbon_intensity_elec'].max()), float(df_neg['carbon_intensity_elec'].mean()))

    # Prediksi
    input_scaled = scaler.transform([[g_in*1e12, p_in*1e6, r_in, c_in]])
    res = model.predict(input_scaled)[0]
    avg_hist = df_neg['primary_energy_consumption'].mean()

    st.markdown("---")
    res_c1, res_c2 = st.columns([1, 2])
    with res_c1:
        st.metric("Hasil Prediksi", f"{res:.2f}", delta=f"{res-avg_hist:.2f} vs Rata-rata")
        st.markdown("### 📝 Interpreter Prediksi")
        if res > avg_hist:
            st.error("Prediksi: Konsumsi Energi Naik. Rekomendasi: Tingkatkan investasi pembangkit baru.")
        else:
            st.success("Prediksi: Konsumsi Efisien. Skenario ini mendukung penghematan energi nasional.")
    with res_c2:
        st.bar_chart(pd.DataFrame({'Data':['Rata-rata', 'Prediksi'], 'Nilai':[avg_hist, res]}), x='Data', y='Nilai')

# ============================================================
# 7️⃣ HALAMAN: PETA ENERGI DUNIA
# ============================================================
elif menu == "🌍 Peta Energi Dunia":
    st.title("🌍 Peta Sebaran Energi")
    avg_energy = df.groupby("country")['primary_energy_consumption'].mean().reset_index()
    
    st.plotly_chart(px.choropleth(avg_energy, locations="country", locationmode="country names", color="primary_energy_consumption", color_continuous_scale="Viridis"), use_container_width=True)

    st.markdown("---")
    st.subheader("📝 Insight Geografis")
    top_c = avg_energy.sort_values('primary_energy_consumption', ascending=False).iloc[0]
    st.info(f"Secara spasial, **{top_c['country']}** mendominasi konsumsi energi. Hal ini menunjukkan pusat aktivitas industri global masih terpusat di wilayah tersebut.")