import pandas as pd
import numpy as np


def generate_sample_data():

    rng = np.random.default_rng(42)

    # 1 year daily data
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")

    # medicines defined as dictionaries (not tuples)
    medicines = [
        {"Product": "Paracetamol 500mg",  "Category": "Analgesic",     "Price": 0.25, "RX": False},
        {"Product": "Ibuprofen 200mg",    "Category": "Analgesic",     "Price": 0.35, "RX": False},
        {"Product": "Cetirizine 10mg",    "Category": "Antihistamine", "Price": 0.30, "RX": False},
        {"Product": "Amoxicillin 500mg",  "Category": "Antibiotic",    "Price": 0.80, "RX": True},
        {"Product": "Azithromycin 250mg", "Category": "Antibiotic",    "Price": 1.10, "RX": True},
        {"Product": "Metformin 500mg",    "Category": "Antidiabetic",  "Price": 0.60, "RX": True},
        {"Product": "Amlodipine 5mg",     "Category": "Cardiac",       "Price": 0.70, "RX": True},
        {"Product": "Omeprazole 20mg",    "Category": "Gastro",        "Price": 0.55, "RX": False},
        {"Product": "Salbutamol Inhaler", "Category": "Respiratory",   "Price": 3.50, "RX": True},
        {"Product": "Vitamin C 1000mg",   "Category": "Supplement",    "Price": 0.40, "RX": False},
    ]

    rows = []

    for i, date in enumerate(dates):

        # seasonal demand pattern
        seasonal = 1 + 0.15 * np.sin(2 * np.pi * i / 365)

        # weekday demand pattern
        weekday_boost = 1.1 if date.weekday() < 5 else 0.9

        demand_factor = seasonal * weekday_boost

        for med in medicines:

            product = med["Product"]
            category = med["Category"]
            base_price = med["Price"]
            rx = med["RX"]

            base_demand = rng.integers(15, 40)

            quantity = int(base_demand * demand_factor + rng.normal(0, 3))
            quantity = max(1, quantity)

            unit_price = round(base_price * rng.uniform(0.95, 1.10), 2)

            revenue = round(quantity * unit_price, 2)

            rows.append({
                "Date": date,
                "Product": product,
                "Category": category,
                "Quantity Sold": quantity,
                "Unit Price": unit_price,
                "Revenue": revenue,
                "Prescription Required": "Yes" if rx else "No",
            })

    df = pd.DataFrame(rows)

    return df


# FIX: module-level code was running every time sidebar.py imported this file
# via importlib, causing generate_sample_data() to run and write a CSV to disk
# silently on every button click. Moved under __main__ guard so it only runs
# when this script is executed directly (e.g. python sample_generator.py).
if __name__ == "__main__":
    data = generate_sample_data()
    print(data.head())
    # optional: save dataset
    data.to_csv("synthetic_pharmacy_sales.csv", index=False)
    print("Saved synthetic_pharmacy_sales.csv")