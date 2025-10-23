from src.data_preparation import load_and_clean_data
from src.data_analysis import descriptive_analysis, diagnostic_analysis
from src.visualization import plot_trend, plot_heatmap
from src.prediction_model import train_predict

def main():
    # 1️⃣ Load & Clean Data
    df = load_and_clean_data("data/owid-energy-data.csv")
    
    # 2️⃣ Analisis Deskriptif & Diagnostik
    descriptive_analysis(df)
    corr = diagnostic_analysis(df)
    
    # 3️⃣ Visualisasi
    plot_trend(df)
    plot_heatmap(corr)
    
    # 4️⃣ Prediksi
    train_predict(df)

if __name__ == "__main__":
    main()
