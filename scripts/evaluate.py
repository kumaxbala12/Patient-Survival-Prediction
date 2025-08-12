#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time_col", default="time")
    p.add_argument("--event_col", default="event")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.test)
    cph = joblib.load(args.model)

    # Risk scores = linear predictor
    risk = cph.predict_partial_hazard(test_df).iloc[:,0].values
    test_df = test_df.copy()
    test_df["risk"] = risk

    # Tertiles of risk
    q1, q2 = np.quantile(risk, [1/3, 2/3])
    def group(r):
        if r <= q1: return "Low"
        if r <= q2: return "Mid"
        return "High"
    test_df["risk_group"] = [group(r) for r in risk]

    # Kaplan-Meier by group
    km = KaplanMeierFitter()
    plt.figure()
    for grp in ["Low","Mid","High"]:
        sub = test_df[test_df["risk_group"] == grp]
        km.fit(sub[args.time_col], sub[args.event_col], label=grp)
        km.plot_survival_function()
    plt.title("KM curves by predicted risk group")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(outdir / "km_by_risk.png", dpi=200)

    # Save groups CSV
    test_df[[args.time_col, args.event_col, "risk", "risk_group"]].to_csv(outdir / "risk_groups.csv", index=False)

    print("Saved evaluation to", outdir)

if __name__ == "__main__":
    main()
