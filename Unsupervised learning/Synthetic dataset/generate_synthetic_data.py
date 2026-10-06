"""
Synthetic customer dataset generator for the Customer Persona Segmentation project.

Creates behavior-based customers from 3 hidden personas, with realistic messiness:
skewed income/spend, spend correlated with purchase count, overlapping groups,
a few missing values and outliers.

Outputs (default: data/raw/):
  customers.csv     -> the features you cluster on (NO persona label)
  ground_truth.csv  -> customer_id + true_persona (only for sanity-checking your clusters)

Usage:
  python generate_synthetic_data.py --n 5000 --seed 42 --out data/raw
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# name, share of customers, and the behavior profile of each hidden persona
PERSONAS = {
    "high_value_loyalist": dict(
        share=0.35, age=(45, 8), income=(11.35, 0.25),      # income ~ lognormal(mu, sigma), median ~ 85k
        purchases=30, aov=(4.9, 0.35), recency=15, web=6,
    ),
    "new_occasional": dict(
        share=0.35, age=(28, 6), income=(10.65, 0.30),      # median ~ 42k
        purchases=4, aov=(4.0, 0.45), recency=40, web=4,
    ),
    "at_risk_dormant": dict(
        share=0.30, age=(50, 10), income=(11.0, 0.30),      # median ~ 60k
        purchases=10, aov=(4.3, 0.40), recency=None, web=1,  # recency handled separately (long gap)
    ),
}


def generate(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = list(PERSONAS)
    shares = np.array([PERSONAS[k]["share"] for k in names])
    persona = rng.choice(names, size=n, p=shares / shares.sum())

    rows = []
    for p in persona:
        cfg = PERSONAS[p]
        age = rng.normal(*cfg["age"])
        income = rng.lognormal(*cfg["income"])
        purchases = rng.poisson(cfg["purchases"])
        aov = rng.lognormal(*cfg["aov"])                     # average order value
        total_spend = purchases * aov
        if cfg["recency"] is None:                           # dormant: last purchase long ago
            recency = rng.normal(180, 50)
        else:
            recency = rng.exponential(cfg["recency"])
        web = rng.poisson(cfg["web"])
        rows.append((age, income, total_spend, purchases, recency, web))

    df = pd.DataFrame(
        rows,
        columns=["age", "income", "total_spend", "num_purchases", "recency_days", "web_visits_per_month"],
    )

    # tidy up ranges / types
    df["age"] = df["age"].clip(18, 85).round().astype(int)
    df["income"] = df["income"].clip(8_000, 400_000).round(2)
    df["total_spend"] = df["total_spend"].clip(0, None).round(2)
    df["recency_days"] = df["recency_days"].clip(0, 365).round().astype(int)

    # realism: a little missing data + a few outliers
    miss_idx = rng.choice(n, size=int(0.015 * n), replace=False)
    df["income"] = df["income"].astype("float")
    df.loc[miss_idx, "income"] = np.nan
    out_idx = rng.choice(n, size=max(3, int(0.003 * n)), replace=False)
    df.loc[out_idx, "total_spend"] *= rng.uniform(5, 10, size=len(out_idx))

    df.insert(0, "customer_id", [f"C{i:05d}" for i in range(1, n + 1)])
    truth = pd.DataFrame({"customer_id": df["customer_id"], "true_persona": persona})
    return df, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    customers, truth = generate(args.n, args.seed)
    customers.to_csv(out / "customers.csv", index=False)
    truth.to_csv(out / "ground_truth.csv", index=False)
    print(f"Wrote {len(customers)} rows to {out}/customers.csv and ground_truth.csv")
