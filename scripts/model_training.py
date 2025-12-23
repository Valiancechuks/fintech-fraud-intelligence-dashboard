import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Loading pseudo-labeled dataset...")

DATA_PATH = "../data/fintech_transactions_pseudo_labeled.csv"

df = pd.read_csv(DATA_PATH)
print("Loaded successfully!")

# -----------------------------
# 1. Feature Selection
# -----------------------------
feature_cols = [
    "amount_log",
    "prev_amount",
    "amount_diff",
    "txn_time_diff_hr",
    "is_weekend",
    "day_of_week",
    "merchant_category_encoded",
    "payment_method_encoded",
    "country_encoded",
    "currency_encoded"
]

print("Using features:")
print(feature_cols)

X = df[feature_cols]
y = df["weak_label"]

# -----------------------------
# 2. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# 3. Model Training
# -----------------------------
print("Training final model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training complete!")

# -----------------------------
# 4. Evaluation
# -----------------------------
print("\nEvaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

roc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc:.4f}")

# -----------------------------
# 5. Feature Importance
# -----------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances_sorted = importances.sort_values(ascending=False)

print("\nFeature Importance:")
print(importances_sorted)

# Save to file for plotting
importances_sorted.to_csv("../data/model_feature_importances.csv")

# -----------------------------
# 6. Save Model
# -----------------------------
MODEL_PATH = "../assets/final_fraud_model.pkl"
os.makedirs("../assets", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training pipeline completed successfully.")


# --- FIX: Save model in compatible pickle format ---
import pickle

model_path = "../assets/final_fraud_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f, protocol=4)

print("Model saved successfully with protocol=4")
