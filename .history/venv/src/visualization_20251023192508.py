import matplotlib.pyplot as plt
import seaborn as sns

def plot_trend(df):
    plt.figure(figsize=(8,5))
    for c in df['country'].unique():
        subset = df[df['country'] == c]
        plt.plot(subset['year'], subset['primary_energy_consumption'], label=c)
    plt.title("Tren Konsumsi Energi per Negara")
    plt.xlabel("Tahun")
    plt.ylabel("Konsumsi Energi (Normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmap(corr):
    plt.figure(figsize=(5,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Korelasi GDP, CO₂, dan Energi")
    plt.tight_layout()
    plt.show()
