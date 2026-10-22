"""Step 2: Optuna-tuned XGBoost (primary model) + SHAP + saved models.
Hyper-parameters are tuned on TRAINING data only (chronological split of fold 1's training set),
then reused for every walk-forward fold - the test weeks are never used for tuning."""
import json, numpy as np, pandas as pd, optuna, shap, joblib
from xgboost import XGBRegressor
from src.data import FEATURES, TARGET, load_folds, load_cnd_forecast
from src.metrics import mape

optuna.logging.set_verbosity(optuna.logging.WARNING)
folds = load_folds(); cnd = load_cnd_forecast()

# ---- 1) tune on training data only (first 80% fit, last 20% validate, chronological) ----
_, tr0, _ = folds[0]
cut = int(len(tr0) * 0.8)
fit, val = tr0.iloc[:cut], tr0.iloc[cut:]

def objective(trial):
    p = dict(n_estimators=trial.suggest_int('n_estimators', 150, 600),
             learning_rate=trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
             max_depth=trial.suggest_int('max_depth', 3, 9),
             subsample=trial.suggest_float('subsample', 0.6, 1.0),
             colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
             min_child_weight=trial.suggest_int('min_child_weight', 1, 10))
    m = XGBRegressor(**p, random_state=42, n_jobs=4).fit(fit[FEATURES], fit[TARGET])
    return mape(val[TARGET], m.predict(val[FEATURES]))

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, timeout=150)
best = study.best_params
print('best params', best, 'val MAPE', round(study.best_value, 3), 'trials', len(study.trials))
json.dump({'best_params': best, 'val_mape': study.best_value, 'n_trials': len(study.trials)},
          open('results/optuna_best.json', 'w'), indent=2)

# ---- 2) walk-forward evaluation with the tuned params + SHAP ----
base = pd.read_csv('results/baseline_predictions.csv', parse_dates=['datetime'])
tuned_preds, shap_rows, shap_vals = [], [], []
for name, tr, te in folds:
    m = XGBRegressor(**best, random_state=42, n_jobs=4).fit(tr[FEATURES], tr[TARGET])
    tuned_preds.append(pd.DataFrame({'datetime': te['datetime'], 'fold': name, 'XGBoost_tuned': m.predict(te[FEATURES])}))
    sv = shap.TreeExplainer(m).shap_values(te[FEATURES])
    shap_vals.append(sv); shap_rows.append(te[FEATURES])
    joblib.dump(m, f"results/xgb_{name.replace(',', '').replace(' ', '_')}.joblib")
T = pd.concat(tuned_preds, ignore_index=True)
out = base.merge(T, on=['datetime', 'fold'])
out.to_csv('results/predictions_all.csv', index=False)
np.save('results/shap_values.npy', np.vstack(shap_vals))
pd.concat(shap_rows).to_csv('results/shap_features.csv', index=False)
print('saved; tuned overall MAPE', round(mape(out[TARGET], out['XGBoost_tuned']), 3))
