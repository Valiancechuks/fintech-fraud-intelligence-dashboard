import os
import joblib
import shap
import pandas as pd
import numpy as np

# ---------------------------------------------------
# 1. Resolve paths
# ---------------------------------------------------
print("Locating model file in assets...")
model_path = os.path.join("..", "assets", "random_forest_clean.pkl")
if not os.path.exists(model_path):
    model_path = os.path.join("assets", "random_forest_clean.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file random_forest_clean.pkl not found.")

print(f"Model found: {model_path}")

print("Locating engineered dataset in data/...")
data_path = os.path.join("..", "data", "fintech_transactions_engineered.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "fintech_transactions_engineered.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError("Engineered dataset fintech_transactions_engineered.csv not found.")

print(f"Engineered dataset found: {data_path}")

# ---------------------------------------------------
# 2. Load model + data
# ---------------------------------------------------
print("Loading model...")
model = joblib.load(model_path)

print("Loading dataset...")
df = pd.read_csv(data_path)
original_cols = list(df.columns)
print(f"Initial columns: {len(original_cols)} -> {original_cols[:10]} ...")

# ---------------------------------------------------
# 3. One-hot encode dataset
# ---------------------------------------------------
print("One-hot encoding using pandas.get_dummies...")
df = pd.get_dummies(df, drop_first=False)

# ---------------------------------------------------
# 4. Align dataframe columns to model expected features
# ---------------------------------------------------
model_features = list(model.feature_names_in_)
print(f"Model.feature_names_in_ length: {len(model_features)}")

# Add missing columns
missing_cols = [col for col in model_features if col not in df.columns]
for col in missing_cols:
    df[col] = 0

# Drop extra columns
extra_cols = [col for col in df.columns if col not in model_features]
if len(extra_cols) > 0:
    print(f"Dropping {len(extra_cols)} extra columns (example: {extra_cols[:6]})")
    df = df.drop(columns=extra_cols)

# Reorder columns
df = df[model_features]

print(f"Final feature matrix shape: {df.shape}")

# ---------------------------------------------------
# 5. Convert to numeric strictly (avoids dtype('O') errors)
# ---------------------------------------------------
print("Converting to numeric matrix to prevent dtype('O') issues...")
X = df.astype("float64").values

# ---------------------------------------------------
# 6. Create safe SHAP background (random 100 rows)
# ---------------------------------------------------
print("Creating SHAP background sample...")
background_size = min(100, X.shape[0])
background = shap.sample(X, background_size)

# ---------------------------------------------------
# 7. Build TreeExplainer safely
# ---------------------------------------------------
print("Building TreeExplainer and computing SHAP values...")

# Use full normalization path to avoid new SHAP errors
explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional",
    model_output="raw"
)

print("Computing SHAP values...")
shap_values = explainer.shap_values(X[:200])  # limit to 200 for speed

print("SHAP computation successful.")

# ---------------------------------------------------
# 8. Save summary plot
# ---------------------------------------------------
print("Generating summary plot...")
shap.summary_plot(shap_values, df.iloc[:200], show=False)
plot_path = "shap_summary.png"

import matplotlib.pyplot as plt
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"SHAP summary plot saved to {plot_path}")
print("DONE.")