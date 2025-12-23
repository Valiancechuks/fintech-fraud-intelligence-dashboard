import pandas as pd
import numpy as np

print("Loading raw dataset...")
df = pd.read_csv("../data/fintech_transactions.csv")
print("Loaded!\n")

# === BASIC CLEANING ===
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

df.sort_values(["customer_id", "date"], inplace=True)

# === FEATURE ENGINEERING ===
df["amount_log"] = np.log1p(df["amount"])

df["prev_amount"] = df.groupby("customer_id")["amount"].shift(1)
df["amount_diff"] = df["amount"] - df["prev_amount"]

df["txn_time_diff_hr"] = (
    df.groupby("customer_id")["date"].diff().dt.total_seconds() / 3600
)

# Fill missing values
df["prev_amount"] = df["prev_amount"].fillna(0)
df["amount_diff"] = df["amount_diff"].fillna(0)
df["txn_time_diff_hr"] = df["txn_time_diff_hr"].fillna(0)

df["day_of_week"] = df["date"].dt.dayofweek

# === CATEGORICAL ENCODING ===
categorical_cols = ["merchant_category", "payment_method", "country", "currency"]

for col in categorical_cols:
    df[col + "_encoded"] = df[col].astype("category").cat.codes

print("\nEngineered columns created!")

df.to_csv("../data/fintech_transactions_engineered.csv", index=False)

print("Saved: fintech_transactions_engineered.csv")