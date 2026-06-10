import pandas as pd
import joblib

from risk_profiles import RISK_PROFILES

model = joblib.load("models/investor_profile_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/features.pkl")

user_data = {
    "age": 30,
    "salary": 50000,
    "monthly_expenses": 2500,
    "savings": 10000,
    "debt": 5000,
    "dependents": 0,
    "investment_horizon": 20,
    "risk_score": 7,
    "savings_rate": (50000 / 12 - 2500) / (50000 / 12),
    "emergency_fund_months": 10000 / 2500,
    "debt_to_income": 5000 / 50000
}

input_df = pd.DataFrame([user_data])
input_df = input_df[features]

prediction_encoded = model.predict(input_df)[0]
profile = label_encoder.inverse_transform([prediction_encoded])[0]

allocation = RISK_PROFILES[profile]

print("Investor Profile:", profile)
print("\nPortfolio Allocation:")

for asset, percentage in allocation.items():
    print(f"{asset}: {percentage}%")