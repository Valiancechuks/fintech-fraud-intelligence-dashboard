import pandas as pd
import pickle
import joblib
import numpy as np

# -------------------------------
# Load Dataset
# -------------------------------
print("Loading pseudo-labeled dataset...")
df = pd.read_csv("../data/fintech_transactions_pseudo_labeled.csv")
print("Loaded successfully!")

feature_cols = [
    'amount_log', 'prev_amount', 'amount_diff', 'txn_time_diff_hr',
    'is_weekend', 'day_of_week', 'merchant_category_encoded',
    'payment_method_encoded', 'country_encoded', 'currency_encoded'
]

X = df[feature_cols]

# -------------------------------
# Load Model (Safe Mode)
# -------------------------------
print("Loading trained model...")

try:
    with open("../assets/final_fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception:
    # fallback – works for some sklearn objects
    model = joblib.load("../assets/final_fraud_model.pkl")

print("Model loaded successfully!")

# -------------------------------
# Make Predictions
# -------------------------------
print("Generating predictions...")
df["fraud_probability"] = model.predict_proba(X)[:, 1]
df["fraud_prediction"] = model.predict(X)

# -------------------------------
# Export for Power BI
# -------------------------------
output_path = "../data/fintech_transactions_analytics.csv"
df.to_csv(output_path, index=False)

print(f"Export completed! File saved to: {output_path}")