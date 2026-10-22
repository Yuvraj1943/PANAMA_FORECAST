"""Reproduction check against the source paper (Aguilar Madrid & Antonio, Information 2021, 12(2), 50).
1) CND's weekly errors on the 14 test weeks are compared with the paper's Table 5.
2) XGBoost is trained with the hyperparameters PUBLISHED in the paper (Table 2, grid-search selection),
   so no settings are chosen by us after seeing the test weeks. Results go to results/paper_params_*.
Also computes the paper's secondary metrics (weekly peak, valley and energy deviation)."""
import json, numpy as np, pandas as pd, joblib, shap
from xgboost import XGBRegressor
from src.data import FEATURES, TARGET, load_folds, load_cnd_forecast
from src.metrics import mape, mae, rmse

PAPER_CND_WEEKLY = [3.90, 3.10, 6.08, 5.55, 4.16, 4.68, 5.04, 6.65, 5.38, 4.06, 3.79, 2.93, 9.06, 4.96]  # Table 5, Pre-Disp. row
PAPER_XGB_AVG = 3.74                                                                                      # Results section: average MAPE of XGB
PARAMS = dict(n_estimators=300, max_depth=7, subsample=0.7, colsample_bytree=0.7, colsample_bylevel=0.7, colsample_bynode=0.7,
              learning_rate=0.045, min_child_weight=7, gamma=15, random_state=123, eval_metric='rmse', n_jobs=2)  # Table 2, GS selection

cnd = load_cnd_forecast(); rows = []; shap_vals = []
for name, tr, te in load_folds():
    te = te.merge(cnd, on='datetime', how='left')
    m = XGBRegressor(**PARAMS).fit(tr[FEATURES], tr[TARGET]); p = m.predict(te[FEATURES]); shap_vals.append(shap.TreeExplainer(m).shap_values(te[FEATURES])); last_model = m; a = te[TARGET].values; c = te['CND'].values
    rows.append(dict(fold=name, CND_MAPE=mape(a, c), PaperParams_MAPE=mape(a, p),
                     CND_peak=abs(a.max() - c.max()) / a.max() * 100, XGB_peak=abs(a.max() - p.max()) / a.max() * 100,
                     CND_valley=abs(a.min() - c.min()) / a.min() * 100, XGB_valley=abs(a.min() - p.min()) / a.min() * 100,
                     CND_energy=abs(a.sum() - c.sum()) / a.sum() * 100, XGB_energy=abs(a.sum() - p.sum()) / a.sum() * 100,
                     CND_bias=float(np.mean(c - a)), XGB_bias=float(np.mean(p - a)), _a=a, _c=c, _p=p, _dt=te['datetime'].values))
R = pd.DataFrame(rows)
np.save('results/shap_values_paper.npy', np.vstack(shap_vals)); joblib.dump(last_model, 'results/xgb_published_settings_last_week.joblib')
A = np.concatenate(R['_a']); C = np.concatenate(R['_c']); P = np.concatenate(R['_p'])
out = {'paper_cnd_weekly': PAPER_CND_WEEKLY, 'our_cnd_weekly': [round(x, 2) for x in R['CND_MAPE']],
       'cnd_max_abs_diff_vs_paper': float(max(abs(a - b) for a, b in zip(PAPER_CND_WEEKLY, R['CND_MAPE']))),
       'paper_xgb_avg_mape': PAPER_XGB_AVG, 'paperparams_xgb_avg_mape': float(R['PaperParams_MAPE'].mean()),
       'cnd_avg_mape': float(R['CND_MAPE'].mean()), 'weekly_xgb_mape': [round(x, 2) for x in R['PaperParams_MAPE']],
       'weeks_xgb_beats_cnd': int((R['PaperParams_MAPE'] < R['CND_MAPE']).sum()),
       'pooled': {'CND': dict(MAPE=mape(A, C), MAE=mae(A, C), RMSE=rmse(A, C)), 'XGBoost_paper_params': dict(MAPE=mape(A, P), MAE=mae(A, P), RMSE=rmse(A, P))},
       'secondary': {k: dict(CND=float(R['CND_' + k].mean()), XGBoost=float(R['XGB_' + k].mean())) for k in ['peak', 'valley', 'energy']},
       'bias_mwh': dict(CND=float(R['CND_bias'].mean()), XGBoost=float(R['XGB_bias'].mean())), 'params': {k: v for k, v in PARAMS.items() if k != 'n_jobs'}}
json.dump(out, open('results/paper_reproduction.json', 'w'), indent=2)
R.drop(columns=['_a', '_c', '_p', '_dt']).round(3).to_csv('results/paper_reproduction_weekly.csv', index=False)
pd.DataFrame({'datetime': np.concatenate(R['_dt']), 'DEMAND': A, 'CND': C, 'XGBoost_paper_params': P}).to_csv('results/predictions_paper_params.csv', index=False)
print(json.dumps({k: out[k] for k in ['cnd_max_abs_diff_vs_paper', 'paper_xgb_avg_mape', 'paperparams_xgb_avg_mape', 'cnd_avg_mape', 'weeks_xgb_beats_cnd', 'pooled', 'secondary', 'bias_mwh']}, indent=1))
print('weekly XGB (paper params):', out['weekly_xgb_mape'])
