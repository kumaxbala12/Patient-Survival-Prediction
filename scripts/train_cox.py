#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
from pathlib import Path
from lifelines import CoxPHFitter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time_col", default="time")
    p.add_argument("--event_col", default="event")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(train_df, duration_col=args.time_col, event_col=args.event_col)
    cph.print_summary()

    # Save model
    joblib.dump(cph, outdir / "coxph.pkl")

    # Save coefficients
    coefs = cph.params_.sort_values(ascending=False)
    coefs.to_csv(outdir / "coefficients.csv")

    # Concordance on train
    c_index = cph.concordance_index_
    pd.Series({"concordance_index_train": c_index}).to_csv(outdir / "metrics_train.csv")

    print("Saved model + metrics to", outdir)

if __name__ == "__main__":
    main()
