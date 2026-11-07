"""Creates sample_sales.csv (2 years of fake retail sales) for testing."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500
dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
regions = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.2, 0.3, 0.2])
products = rng.choice(["Laptop", "Phone", "Tablet", "Headphones", "Monitor"], n, p=[0.15, 0.3, 0.15, 0.25, 0.15])
price = {"Laptop": 800, "Phone": 500, "Tablet": 350, "Headphones": 80, "Monitor": 220}
qty = rng.integers(1, 6, n)
unit = np.array([price[p] for p in products]) * rng.uniform(0.9, 1.1, n)
season = 1 + 0.25 * np.sin((dates.month - 3) / 12 * 2 * np.pi) + (dates.month == 11) * 0.3
df = pd.DataFrame({
    "order_id": np.arange(1001, 1001 + n),
    "order_date": dates.strftime("%Y-%m-%d"),
    "region": regions,
    "product": products,
    "quantity": qty,
    "unit_price": unit.round(2),
    "sales": (qty * unit * season).round(2),
    "customer_rating": rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.04, 0.08, 0.18, 0.35, 0.3, 0.05]),
}).sort_values("order_date")
df.to_csv("sample_sales.csv", index=False)
print("Created sample_sales.csv", df.shape)
