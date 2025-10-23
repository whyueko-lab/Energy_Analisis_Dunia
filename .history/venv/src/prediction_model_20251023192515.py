from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_predict(df):
    # Fitur dan target
    X = df[['gdp', 'population', 'renewables_share_energy', 'carbon_intensity_elec']]
    y = df['primary_energy_consumption']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    print(f"\n🤖 Mean Squared Error (MSE): {mse:.4f}")
    return model
