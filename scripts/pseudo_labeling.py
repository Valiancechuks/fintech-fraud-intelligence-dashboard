import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

print("Loading engineered dataset...")
df = pd.read_csv("../data/fintech_transactions_engineered.csv")
print("Loaded successfully!")

# -------------------------------------------------------------
# Inspect columns
# -------------------------------------------------------------
cols = df.columns.tolist()
print("Columns:", cols)

# -------------------------------------------------------------
# Create is_weekend if missing
# -------------------------------------------------------------
if "day_of_week" not in df.columns:
    raise ValueError("day_of_week not found in engineered dataset")

df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# -------------------------------------------------------------
# Weak labels
# -------------------------------------------------------------
amount_diff_thr = df["amount_diff"].quantile(0.90)
rapid_tx_thr = df["txn_time_diff_hr"].quantile(0.10)
high_amount_thr = df["amount_log"].quantile(0.90)

df["weak_label"] = (
    (df["amount_diff"] > amount_diff_thr) |
    (df["txn_time_diff_hr"] < rapid_tx_thr) |
    ((df["is_weekend"] == 1) & (df["amount_log"] > high_amount_thr))
).astype(int)

print("Weak labels created.")

# -------------------------------------------------------------
# Features that MUST exist
# -------------------------------------------------------------
base_features = [
    "amount_log",
    "prev_amount",
    "amount_diff",
    "txn_time_diff_hr",
    "is_weekend",
    "day_of_week"
]

# Optional features (include only if present)
optional_features = [
    "merchant_category_encoded",
    "payment_method_encoded",
    "country_encoded",
    "currency_encoded",
]

feature_cols = [c for c in base_features if c in df.columns] + \
               [c for c in optional_features if c in df.columns]

print("Using features:", feature_cols)

X = df[feature_cols]
y = df["weak_label"]

# -------------------------------------------------------------
# Train base model
# -------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1
}

model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=200)
print("Base model trained.")

# -------------------------------------------------------------
# Pseudo-labeling
# -------------------------------------------------------------
df["fraud_label"] = (model.predict(X) > 0.5).astype(int)

df.to_csv("../data/fintech_transactions_pseudo_labeled.csv", index=False)
print("Saved: ../data/fintech_transactions_pseudo_labeled.csv")

print(df["fraud_label"].value_counts())