#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--survival", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id_col", default="id")
    p.add_argument("--time_col", default="time")
    p.add_argument("--event_col", default="event")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(args.features)
    y = pd.read_csv(args.survival)

    if args.id_col not in X.columns or args.id_col not in y.columns:
        raise ValueError("id_col must exist in both files")

    df = y.merge(X, on=args.id_col, how="inner")
    if df.empty:
        raise ValueError("No overlapping IDs between features and survival")

    numeric_cols = [c for c in df.columns if c not in [args.id_col, args.time_col, args.event_col]]
    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Z = imputer.fit_transform(df[numeric_cols].astype(float))
    Z = scaler.fit_transform(Z)

    proc = pd.DataFrame(Z, columns=numeric_cols, index=df.index)
    proc[args.id_col] = df[args.id_col].values
    proc[args.time_col] = df[args.time_col].astype(float).values
    proc[args.event_col] = df[args.event_col].astype(int).values

    train_df, test_df = train_test_split(proc, test_size=args.test_size, random_state=args.random_state)

    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    # Save metadata (column order)
    meta = {
        "id_col": args.id_col,
        "time_col": args.time_col,
        "event_col": args.event_col,
        "feature_cols": numeric_cols
    }
    pd.Series(meta).to_json(out / "preprocess_meta.json")

    print("Saved to", out)

if __name__ == "__main__":
    main()
