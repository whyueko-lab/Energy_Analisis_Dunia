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

    # Pilih Mode Analisis
    mode = st.radio("Pilih Mode:", ["Analisis Satu Negara", "Perbandingan Dua Negara"])

    # ================================================
    # MODE 1: ANALISIS SATU NEGARA
    # ================================================
    if mode == "Analisis Satu Negara":
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

        st.markdown("""
        ### 📘 Penjelasan Korelasi
        Korelasi menunjukkan hubungan antar-variabel:  
        - Nilai positif tinggi menunjukkan dua variabel naik bersama.  
        - Nilai negatif tinggi menunjukkan saling berlawanan.  
        - Nilai mendekati 0 berarti hubungan lemah.  
        """)
        
        # ====================================
        # INTERPRETASI OTOMATIS – SATU NEGARA
        # ====================================
        st.subheader("🧠 Interpretasi Otomatis")

        # Hitung rata-rata
        gdp_mean = data_negara['gdp'].mean()
        energy_mean = data_negara['primary_energy_consumption'].mean()
        carbon_mean = data_negara['carbon_intensity_elec'].mean()
        renew_mean = data_negara['renewables_share_energy'].mean()

        # Hitung tren (kenaikan / penurunan)
        data_sorted = data_negara.sort_values('year')
        energy_trend = data_sorted['primary_energy_consumption'].diff().mean()
        carbon_trend = data_sorted['carbon_intensity_elec'].diff().mean()
        renew_trend = data_sorted['renewables_share_energy'].diff().mean()

        def trend_text(value, label):
            if value > 0:
                return f"• **{label} menunjukkan kecenderungan naik** sepanjang tahun."
            elif value < 0:
                return f"• **{label} menunjukkan kecenderungan turun**, tanda perubahan positif."
            else:
                return f"• **{label} relatif stabil** tanpa perubahan signifikan."

        # Interpretasi korelasi
        corr_gdp_energy = corr.loc['gdp','primary_energy_consumption']
        corr_energy_carbon = corr.loc['primary_energy_consumption','carbon_intensity_elec']

        def interpret_corr(value, var1, var2):
            abs_val = abs(value)
            if abs_val > 0.7:
                strength = "hubungan kuat"
            elif abs_val > 0.4:
                strength = "hubungan sedang"
            else:
                strength = "hubungan lemah"

            if value > 0:
                direction = "bergerak searah"
            elif value < 0:
                direction = "bergerak saling berlawanan"
            else:
                direction = "tidak punya arah hubungan"

            return f"• **{var1} dan {var2}** memiliki **{strength}** dan cenderung **{direction}**."

        st.markdown(f"""
        ### 📌 Ringkasan Energi untuk **{negara}**
        • Rata-rata konsumsi energi: **{energy_mean:.3f}**  
        • Intensitas karbon listrik: **{carbon_mean:.3f}**  
        • Proporsi energi terbarukan: **{renew_mean:.3f}**  
        • Aktivitas ekonomi (GDP): **{gdp_mean:.3f}**

        ### 📈 Tren dari Waktu ke Waktu
        {trend_text(energy_trend, "Konsumsi energi")}
        {trend_text(carbon_trend, "Intensitas karbon listrik")}
        {trend_text(renew_trend, "Energi terbarukan")}

        ### 🔍 Interpretasi Korelasi
        {interpret_corr(corr_gdp_energy, "GDP", "konsumsi energi")}
        {interpret_corr(corr_energy_carbon, "konsumsi energi", "intensitas karbon")}
        """)    
        

    # ================================================
    # MODE 2: PERBANDINGAN DUA NEGARA
    # ================================================
    else:
        st.subheader("🔍 Perbandingan Dua Negara")

        col1, col2 = st.columns(2)
        with col1:
            negara1 = st.selectbox("Pilih Negara 1:", sorted(df['country'].unique()), key="n1")
        with col2:
            negara2 = st.selectbox("Pilih Negara 2:", sorted(df['country'].unique()), key="n2")

        data1 = df[df['country'] == negara1]
        data2 = df[df['country'] == negara2]

        st.markdown(f"### 📊 Ringkasan Statistik\nPerbandingan antara **{negara1}** dan **{negara2}**")

        # Tampilkan mean tiap variabel
        compare_df = pd.DataFrame({
            'Variabel': ['GDP', 'Populasi', 'Konsumsi Energi', 'Energi Terbarukan', 'Intensitas Karbon'],
            negara1: [
                data1['gdp'].mean(),
                data1['population'].mean(),
                data1['primary_energy_consumption'].mean(),
                data1['renewables_share_energy'].mean(),
                data1['carbon_intensity_elec'].mean()
            ],
            negara2: [
                data2['gdp'].mean(),
                data2['population'].mean(),
                data2['primary_energy_consumption'].mean(),
                data2['renewables_share_energy'].mean(),
                data2['carbon_intensity_elec'].mean()
            ],
        })

        st.dataframe(compare_df)

        # Heatmap dua negara
        st.subheader("📈 Korelasi Masing-Masing Negara")
        colA, colB = st.columns(2)

        with colA:
            st.markdown(f"**{negara1}**")
            corr1 = data1[['gdp', 'primary_energy_consumption', 'carbon_intensity_elec']].corr()
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            sns.heatmap(corr1, annot=True, cmap="coolwarm", ax=ax1)
            st.pyplot(fig1)

        with colB:
            st.markdown(f"**{negara2}**")
            corr2 = data2[['gdp', 'primary_energy_consumption', 'carbon_intensity_elec']].corr()
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.heatmap(corr2, annot=True, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)

        # ==========================================
        # INTERPRETASI OTOMATIS
        # ==========================================
        st.subheader("🧠 Interpretasi Otomatis")

        def interpret(var, label):
            val1 = compare_df[negara1][compare_df['Variabel'] == var].values[0]
            val2 = compare_df[negara2][compare_df['Variabel'] == var].values[0]

            if val1 > val2:
                return f"• **{negara1}** memiliki {label} lebih tinggi dibanding **{negara2}**."
            elif val2 > val1:
                return f"• **{negara2}** memiliki {label} lebih tinggi dibanding **{negara1}**."
            else:
                return f"• Kedua negara memiliki {label} yang hampir sama."

        st.markdown(f"""
        #### 📌 Kesimpulan Perbandingan:
        {interpret("GDP", "GDP")}
        {interpret("Populasi", "jumlah penduduk")}
        {interpret("Konsumsi Energi", "konsumsi energi")}
        {interpret("Energi Terbarukan", "porsi energi terbarukan")}
        {interpret("Intensitas Karbon", "intensitas karbon listrik")}
        """)
        
        

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
