import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------
# Paths
# --------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "fintech_transactions.csv")
ASSETS_DIR = os.path.join(BASE_DIR, "assets", "eda_charts")

os.makedirs(ASSETS_DIR, exist_ok=True)

# --------------------------------------
# Load Data
# --------------------------------------
print("\n Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("\n Dataset Loaded Successfully!")
print(df.head())

# --------------------------------------
# Basic Info
# --------------------------------------
print("\n Dataset Info:")
print(df.info())

print("\n Missing Values:")
print(df.isnull().sum())

print("\n Summary Statistics (Numerical):")
print(df.describe())

# --------------------------------------
# Categorical distributions
# --------------------------------------
categoricals = ["payment_method", "status", "country", "currency", "merchant_category", "card_provider"]

for col in categoricals:
    print(f"\n {col.upper()} VALUE COUNTS:")
    print(df[col].value_counts(dropna=False))

# --------------------------------------
# Status Distribution Plot
# --------------------------------------
plt.figure(figsize=(7,5))
df["status"].value_counts().plot(kind="bar")
plt.title("Transaction Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "status_distribution.png"))
plt.close()

# --------------------------------------
# Payment Method Distribution
# --------------------------------------
plt.figure(figsize=(7,5))
df["payment_method"].value_counts().plot(kind="bar")
plt.title("Payment Method Distribution")
plt.xlabel("Payment Method")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "payment_method_distribution.png"))
plt.close()

# --------------------------------------
# Amount Distribution
# --------------------------------------
plt.figure(figsize=(7,5))
df["amount"].plot(kind="hist", bins=40)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "amount_distribution.png"))
plt.close()

# --------------------------------------
# Country Analysis
# --------------------------------------
plt.figure(figsize=(7,5))
df["country"].value_counts().plot(kind="bar")
plt.title("Transactions by Country")
plt.xlabel("Country")
plt.ylabel("Volume")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "country_distribution.png"))
plt.close()

# --------------------------------------
# Integrity Checks
# --------------------------------------

print("\n Integrity Check: Currency per Country")
currency_map_errors = df.groupby(["country", "currency"]).size()
print(currency_map_errors)

print("\n Integrity Check: card_last_4 should be null when payment is not Card/POS")
invalid_card = df[
    (~df["payment_method"].isin(["Card", "POS"])) &
    (df["card_last_4"].notnull())
]

print("Invalid card_last_4 rows:", len(invalid_card))

print("\n Integrity Check Completed.")
print("\n EDA Completed Successfully!")
print(f"Charts saved to: {ASSETS_DIR}")
