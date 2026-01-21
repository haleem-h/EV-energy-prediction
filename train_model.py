import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/ev_charging_sessions.csv")
data.dropna(inplace=True)

# Convert time columns
data['Charging_Start_Time'] = pd.to_datetime(data['Charging_Start_Time'])
data['Charging_End_Time'] = pd.to_datetime(data['Charging_End_Time'])

# Create charging duration
data['Charging_Duration_Min'] = (
    data['Charging_End_Time'] - data['Charging_Start_Time']
).dt.total_seconds() / 60

# Create target variable
data['Energy_Level'] = pd.cut(
    data['Energy_Consumed_kWh'],
    bins=[0, 10, 30, data['Energy_Consumed_kWh'].max()],
    labels=['Low', 'Medium', 'High']
)

# Encode categorical columns
encoder = LabelEncoder()
data['Vehicle_Type'] = encoder.fit_transform(data['Vehicle_Type'])
data['Payment_Method'] = encoder.fit_transform(data['Payment_Method'])

# Select features and target
X = data[['Vehicle_Type', 'Charging_Duration_Min', 'Cost_INR', 'Payment_Method']]
y = data['Energy_Level']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
import os

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/energy_model.pkl")

print("Model trained and saved successfully")
