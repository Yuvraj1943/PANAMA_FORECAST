#!/usr/bin/env bash
# Re-creates every result from scratch (needs the 4 data files in data/). Takes roughly 15-20 minutes on one CPU.
set -e
python run_baselines.py      # persistence, linear regression, XGBoost on the 14 walk-forward folds
python run_xgb_tuned.py      # Optuna tuning (training data only), tuned XGBoost, SHAP values, saved models
python run_lstm.py           # compact LSTM on the same folds (the slowest step)
python run_reproduction.py   # XGBoost with the settings PUBLISHED in the source paper; checks CND weekly errors against its Table 5
python run_analysis.py       # metrics overall / per week / holiday / COVID, charts
python make_slide_figures.py  # title-slide chart and architecture diagram used in the slides
python -m pytest -q          # QA test-suite
