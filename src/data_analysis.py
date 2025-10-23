def descriptive_analysis(df):
    print("\n📘 Ringkasan Statistik:")
    print(df.describe())

def diagnostic_analysis(df):
    corr = df[['gdp', 'primary_energy_consumption', 'carbon_intensity_elec']].corr()
    print("\n🔍 Korelasi GDP, Energi, dan CO₂:")
    print(corr)
    return corr
