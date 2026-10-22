"""Step 4: combine predictions, compute metrics (overall / per fold / holiday / COVID), make charts."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap
from src.data import FEATURES, TARGET
from src.metrics import summary, mape

P = pd.read_csv('results/predictions_all.csv', parse_dates=['datetime'])
L = pd.read_csv('results/lstm_predictions.csv', parse_dates=['datetime'])
P = P.merge(L, on=['datetime', 'fold'])
Q = pd.read_csv('results/predictions_paper_params.csv', parse_dates=['datetime'])[['datetime', 'XGBoost_paper_params']].rename(columns={'XGBoost_paper_params': 'XGBoost_paper'})
P = P.merge(Q, on='datetime', how='left')
P.to_csv('results/predictions_final.csv', index=False)
MODELS = {'CND': 'CND (official)', 'Persistence(wk-2)': 'Persistence (2 wks ago)', 'LinearRegression': 'Linear Regression',
          'XGBoost_paper': 'XGBoost (published settings)', 'XGBoost': 'XGBoost (hand-set)', 'XGBoost_tuned': 'XGBoost (Optuna-tuned)', 'LSTM': 'LSTM'}
order = list(P['fold'].unique())
covid_folds = [f for f in order if '2020' in f and ('May' in f or 'Jun' in f)]
P['segment'] = np.where(P['fold'].isin(covid_folds), 'COVID-affected weeks',
                np.where(P['holiday'] == 1, 'Public-holiday hours', 'Ordinary hours'))

overall = pd.DataFrame({MODELS[m]: summary(P[TARGET], P[m]) for m in MODELS}).T.round(3)
seg = {s: pd.DataFrame({MODELS[m]: summary(d[TARGET], d[m]) for m in MODELS}).T.round(2)
       for s, d in P.groupby('segment')}
perfold = P.groupby('fold', sort=False).apply(lambda d: pd.Series({MODELS[m]: mape(d[TARGET], d[m]) for m in MODELS})).round(2)
wins = {MODELS[m]: int((perfold[MODELS[m]] < perfold[MODELS['CND']]).sum()) for m in MODELS if m != 'CND'}
res = {'overall': overall.to_dict('index'), 'segments': {k: v.to_dict('index') for k, v in seg.items()},
       'segment_hours': P['segment'].value_counts().to_dict(), 'folds_better_than_CND': wins,
       'perfold_MAPE': perfold.to_dict('index'), 'covid_folds': covid_folds}
json.dump(res, open('results/summary.json', 'w'), indent=2)
print(overall); print(); [print(k, '\n', v, '\n') for k, v in seg.items()]; print(perfold); print(wins); print(P['segment'].value_counts())

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
# 1 overall bar
fig, ax = plt.subplots(figsize=(7, 3.6)); o = overall['MAPE'].sort_values(ascending=False)
cols = ['#c0392b' if i == 'CND (official)' else '#2c7fb8' if 'XGBoost' in i else '#7f8c8d' for i in o.index]
ax.barh(o.index, o.values, color=cols); [ax.text(v + 0.05, i, f'{v:.2f}%', va='center') for i, v in enumerate(o.values)]
ax.set_xlabel('MAPE (%) - lower is better'); ax.set_title('Overall error across 14 test weeks (2,352 hours)'); plt.tight_layout(); plt.savefig('figures/overall_mape.png', dpi=200); plt.close()
# 2 per fold
fig, ax = plt.subplots(figsize=(8, 3.8)); x = range(len(order))
for m, c, ls in [('CND (official)', '#c0392b', '-'), ('XGBoost (published settings)', '#0F766E', '-'), ('LSTM', '#e67e22', '--')]:
    ax.plot(x, perfold[m].values, marker='o', color=c, ls=ls, label=m)
ax.set_xticks(list(x)); ax.set_xticklabels([f.replace(', ', '\n') for f in order], fontsize=6.5); ax.set_ylabel('MAPE (%)')
ax.set_title('Error per test week'); ax.legend(frameon=False); plt.tight_layout(); plt.savefig('figures/per_fold_mape.png', dpi=200); plt.close()
# 3 example weeks
for fold, fn in [('Week 51, Dec 2019', 'week_dec2019'), ('Week 20, May 2020', 'week_may2020')]:
    d = P[P['fold'] == fold]; fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(d['datetime'], d[TARGET], color='black', lw=1.6, label='Actual demand')
    ax.plot(d['datetime'], d['CND'], color='#c0392b', lw=1.1, label='CND forecast'); ax.plot(d['datetime'], d['XGBoost'], color='#2c7fb8', lw=1.1, label='XGBoost')
    ax.set_ylabel('Demand (MWh)'); ax.set_title(f'{fold}: actual vs forecasts'); ax.legend(frameon=False, ncol=3); plt.xticks(rotation=20, fontsize=8); plt.tight_layout(); plt.savefig(f'figures/{fn}.png', dpi=200); plt.close()
# 4 segments
fig, ax = plt.subplots(figsize=(7.5, 3.6)); segs = ['Ordinary hours', 'Public-holiday hours', 'COVID-affected weeks']; w = 0.27
for k, (m, c) in enumerate([('CND (official)', '#c0392b'), ('XGBoost (published settings)', '#0F766E'), ('LSTM', '#e67e22')]):
    vals = [seg[s].loc[m, 'MAPE'] for s in segs]; ax.bar(np.arange(3) + (k - 1) * w, vals, w, color=c, label=m)
ax.set_xticks(range(3)); ax.set_xticklabels(segs); ax.set_ylabel('MAPE (%)'); ax.set_title('Error by situation'); ax.legend(frameon=False); plt.tight_layout(); plt.savefig('figures/segments.png', dpi=200); plt.close()
# 5 shap
import os
sv = np.load('results/shap_values_paper.npy' if os.path.exists('results/shap_values_paper.npy') else 'results/shap_values.npy'); Xs = pd.read_csv('results/shap_features.csv')
imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(); imp.to_csv('results/shap_importance.csv')
fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.barh(imp.index, imp.values, color='#2c7fb8'); ax.set_xlabel('Mean |SHAP value| (MWh)'); ax.set_title('What drives the XGBoost forecast'); plt.tight_layout(); plt.savefig('figures/shap_importance.png', dpi=200); plt.close()
shap.summary_plot(sv, Xs, show=False, plot_size=(7, 4)); plt.tight_layout(); plt.savefig('figures/shap_beeswarm.png', dpi=200, bbox_inches='tight'); plt.close()
print(imp.sort_values(ascending=False).round(2))
