# Patient Survival Prediction (Cox PH + Kaplan–Meier)

Predict and explain patient survival using clinical/genomic features.

## Dataset (put in `data/raw/`)
Use a Kaggle survival dataset (e.g., breast cancer or TCGA-like). Expected files:
- `features.csv` — rows=samples; must include an `id` column and numeric features
- `survival.csv` — columns: `id`, `time` (follow-up duration), `event` (1=death/endpoint, 0=censored)

You can rename columns in the notebook or pass args to the scripts.

## What’s included
- **Notebook**: `notebooks/01_km_cox_pipeline.ipynb` — EDA, KM curves by risk groups, Cox model, concordance index, risk scores, calibration-style plots
- **Scripts**:
  - `scripts/preprocess.py` — merge, impute, scale; saves train/test
  - `scripts/train_cox.py` — CoxPH model (lifelines), saves model + coefficients
  - `scripts/evaluate.py` — C-index, KM curves for risk tertiles, and figures

## Quickstart
```bash
pip install -r requirements.txt

# 1) Preprocess
python scripts/preprocess.py   --features data/raw/features.csv   --survival data/raw/survival.csv   --out data/processed

# 2) Train
python scripts/train_cox.py   --train data/processed/train.csv   --out results/models

# 3) Evaluate
python scripts/evaluate.py   --test data/processed/test.csv   --model results/models/coxph.pkl   --out results/figures
```
