"""Step 1: walk-forward baselines vs CND's official forecast across the 14 folds."""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from src.data import FEATURES, TARGET, load_folds, load_cnd_forecast
from src.metrics import summary

cnd = load_cnd_forecast()
folds = load_folds()
rows, preds = [], []
for name, tr, te in folds:
    te = te.merge(cnd, on='datetime', how='left')
    out = te[['datetime', TARGET, 'CND']].copy()
    out['fold'] = name
    out['holiday'] = te['holiday'].values
    out['Persistence(wk-2)'] = te['week_X-2'].values
    lr = LinearRegression().fit(tr[FEATURES], tr[TARGET])
    out['LinearRegression'] = lr.predict(te[FEATURES])
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9,
                       colsample_bytree=0.9, random_state=42, n_jobs=4).fit(tr[FEATURES], tr[TARGET])
    out['XGBoost'] = xgb.predict(te[FEATURES])
    preds.append(out)
P = pd.concat(preds, ignore_index=True)
P.to_csv('results/baseline_predictions.csv', index=False)
models = ['CND', 'Persistence(wk-2)', 'LinearRegression', 'XGBoost']
tab = pd.DataFrame({m: summary(P[TARGET], P[m]) for m in models}).T.round(3)
print('OVERALL (14 folds x 168h =', len(P), 'hours)'); print(tab)
per = P.groupby('fold', sort=False).apply(lambda d: pd.Series({m + '_MAPE': summary(d[TARGET], d[m])['MAPE'] for m in models})).round(2)
print(); print(per)
