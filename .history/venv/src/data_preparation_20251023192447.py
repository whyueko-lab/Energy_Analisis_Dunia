import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(path):
    # Membaca file CSV/Excel dari Our World in Data
    df = pd.read_csv(path)
    
    # Pilih variabel yang dibutuhkan
    cols = ['country', 'year', 'primary_energy_consumption',
            'renewables_share_energy', 'carbon_intensity_elec',
            'gdp', 'population']
    df = df[cols]
    
    # Bersihkan data
    df = df.dropna()
    
    # Fokus pada beberapa negara besar
    negara_fokus = ['China', 'United States', 'India', 'Indonesia', 'Brazil']
    df = df[df['country'].isin(negara_fokus)]
    
    # Format kolom
    df['year'] = df['year'].astype(int)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Normalisasi
    scaler = MinMaxScaler()
    numeric_cols = ['primary_energy_consumption', 'renewables_share_energy',
                    'carbon_intensity_elec', 'gdp', 'population']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("✅ Data berhasil dibersihkan dan dinormalisasi.")
    return df
