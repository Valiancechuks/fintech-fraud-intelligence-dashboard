# fintech_transactions.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# -------- CONFIG --------
NUM_ROWS = 10000
OUT_FILE = "fintech_transactions.csv"

# -------- HELPERS --------
def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randrange(delta.days + 1))

def gen_card_number():
    digits = ''.join(random.choice('0123456789') for _ in range(16))
    return f"{digits[0:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:16]}"

def last4(cardnum):
    if not cardnum:
        return None
    return cardnum.replace("-", "")[-4:]

# -------- DOMAINS --------
countries = ["Nigeria", "Kenya", "Ghana", "South Africa", "Uganda"]
currency_map = {
    "Nigeria": "NGN",
    "Kenya": "KES",
    "Ghana": "GHS",
    "South Africa": "ZAR",
    "Uganda": "UGX"
}

card_providers_by_country = {
    "Nigeria": ["Visa", "Mastercard", "Verve"],
    "Kenya": ["Visa", "Mastercard"],
    "Ghana": ["Visa", "Mastercard"],
    "South Africa": ["Visa", "Mastercard"],
    "Uganda": ["Visa", "Mastercard"],
}

payment_methods = ["Card", "Bank Transfer", "USSD", "Wallet", "POS"]
merchant_categories = [
    "E-commerce", "Bills Payment", "Food Delivery",
    "Transportation", "Airtime/Data", "Retail",
    "Professional Services"
]

start_dt = datetime(2015, 1, 1)
end_dt = datetime(2025, 12, 1)

random.seed(42)
np.random.seed(42)

rows = []

# -------- GENERATION --------
for i in range(NUM_ROWS):
    txn_id = f"TXN{100000 + i}"
    dt = random_date(start_dt, end_dt)

    country = random.choices(
        countries,
        weights=[0.45, 0.15, 0.12, 0.18, 0.10]
    )[0]
    currency = currency_map[country]

    payment_method = random.choices(
        payment_methods,
        weights=[0.35, 0.15, 0.15, 0.20, 0.15]
    )[0]

    card_applicable = (payment_method == "Card") or (
        payment_method == "POS" and random.random() < 0.25
    )

    if card_applicable:
        provider = random.choice(card_providers_by_country[country])
        # Ensure Verve stays in Nigeria only
        if provider == "Verve" and country != "Nigeria":
            provider = random.choice([p for p in card_providers_by_country[country] if p != "Verve"])

        # Generate full card internally, but do NOT save it
        full_card = gen_card_number()
        last4_digits = last4(full_card)

    else:
        provider = None
        last4_digits = None

    amount = int(np.clip(np.random.exponential(scale=20000) * 20, 500, 500000))

    # Status generation (simple realistic logic)
    status = random.choices(
        ["Success", "Failed", "Pending", "Reversed"],
        weights=[0.70, 0.15, 0.08, 0.07]
    )[0]

    merchant_category = random.choice(merchant_categories)
    customer_id = f"CUST{random.randint(1000, 9999)}"

    rows.append({
        "transaction_id": txn_id,
        "date": dt.strftime("%Y-%m-%d"),
        "customer_id": customer_id,
        "payment_method": payment_method,
        "card_provider": provider,
        "amount": amount,
        "status": status,
        "merchant_category": merchant_category,
        "country": country,
        "currency": currency,
        "card_last_4": last4_digits
    })

# -------- SAVE --------
df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)
print(f"Saved: {OUT_FILE}")
print(df.head())