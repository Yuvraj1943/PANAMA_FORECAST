# Panama short-term electricity load forecasting (ICT30016 Innovation Project)

Predicts Panama's national hourly electricity demand one week (168 hours) ahead and compares it with the grid
operator CND's own official weekly forecast. Main model: XGBoost with the hyperparameters published by
Aguilar Madrid and Antonio (2021). Secondary model: LSTM. Includes SHAP explainability, a Streamlit dashboard and an
automated test-suite (40 tests).

## Results (14 walk-forward test weeks, 2,352 hours, same hours for every model)
| Model | MAPE % | MAE (MWh) | RMSE (MWh) |
|---|---|---|---|
| XGBoost (published settings) | 3.83 | 46.2 | 57.6 |
| XGBoost (hand-set) | 3.84 | 46.3 | 57.7 |
| XGBoost (Optuna-tuned) | 3.92 | 47.0 | 58.6 |
| LSTM | 4.46 | 53.4 | 69.3 |
| Linear Regression | 4.48 | 53.2 | 71.7 |
| CND (official) | 4.95 | 59.2 | 74.0 |
| Persistence (2 wks ago) | 5.84 | 71.6 | 104.1 |

"XGBoost (published settings)" beats CND in 14 of 14 weeks. "Hand-set" was our own first guess and "Optuna-tuned" is our own search
(20 trials, tuned on training data only); both are reported for transparency.

By situation (MAPE %):
| Situation | Hours | CND | XGBoost (published settings) | LSTM |
|---|---|---|---|---|
| Ordinary hours | 1823 | 4.33 | 3.50 | 3.59 |
| Public-holiday hours | 193 | 7.27 | 4.74 | 5.67 |
| COVID-affected weeks | 336 | 7.01 | 5.10 | 8.52 |

"COVID-affected weeks" = the May and June 2020 test weeks. The paper's secondary weekly metrics (our calculation): peak
2.49% vs 2.75% for CND, valley 3.81% vs 4.49%, energy 1.90% vs 2.81%.

## Checked against the source paper
| Check | Paper [1] | This project |
|---|---|---|
| CND weekly errors, 14 test weeks (Table 5) | 3.90, 3.10, 6.08, 5.55, 4.16, 4.68, 5.04, 6.65, 5.38, 4.06, 3.79, 2.93, 9.06, 4.96 | identical to 2 decimals (max difference 0.005) |
| CND average error | 4.95% | 4.95% |
| XGBoost average error | 3.74% | 3.83% |
| Weeks XGBoost beats CND | 14 of 14 | 14 of 14 |

`run_reproduction.py` does this check; `tests/test_qa.py` repeats it automatically.

## Quick start
```
pip install -r requirements.txt
streamlit run app.py          # dashboard - needs only results/ and figures/ (included)
```
To re-train or run the tests, first put the four data files in `data/` (see `data/README.txt`), then:
```
python -m pytest -q           # 40 tests
./run_all.sh                  # rebuild every result (about 15-20 min on one CPU)
```

## Folder guide
- `src/` data loading and metric definitions (MAPE, MAE, RMSE)
- `run_baselines.py`, `run_xgb_tuned.py`, `run_lstm.py`, `run_reproduction.py`, `run_analysis.py` the pipeline steps
- `app.py` Streamlit dashboard: forecast viewer, upload and predict, evaluation, explainability
- `tests/test_qa.py` data integrity, 72-hour leakage gap, metric formulas, model sanity, reproducibility, match with the paper
- `results/` predictions, metrics (summary.json, paper_reproduction.json), SHAP values, saved XGBoost models
- `figures/` charts and dashboard screenshots used in the presentation
- `make_slide_figures.py` draws the title-slide chart and architecture diagram used in the presentation
- `.streamlit/config.toml` dashboard colours and settings
- `docs/pres_final.pdf` a PDF copy of the presentation slides

## Method in brief
Each test week is predicted by a model trained only on data available before that week's cutoff, with the same
72-hour information gap CND works with. Features: demand 2, 3 and 4 weeks earlier, 4-week moving average, hour,
weekday, weekend, holiday flags and Tocumen temperature (the 10 pre-built regressors supplied with the dataset).

## Known limitations
- Only 14 test weeks (2,352 hours); holiday (193 h) and COVID (336 h) groups are small.
- Temperature is observed, not forecast, so real-world accuracy would be somewhat worse.
- Only Tocumen temperature is in the supplied feature set.
- The LSTM is a compact version (last 2 years of training data, 8 epochs, no tuning) built to run on one CPU.
  It is not evidence about the best possible LSTM. GRU, TCN and Transformer models were not tried.
- Not deployed anywhere; the dashboard is a local prototype.
- The dataset's own licence has not been confirmed; check the Mendeley Data page before any public release.

## References
[1] E. Aguilar Madrid and N. Antonio, "Short-term electricity load forecasting with machine learning," Information, vol. 12, no. 2, art. 50, 2021, doi: 10.3390/info12020050.
[2] B. F. Hobbs et al., "Analysis of the value for unit commitment of improved load forecasts," IEEE Trans. Power Syst., vol. 14, no. 4, pp. 1342-1348, 1999, doi: 10.1109/59.801894.
[3] Global Energy Monitor, "Energy profile: Panama," Global Energy Monitor Wiki.
[4] IRENA, "Energy profile: Panama," Statistical profiles.
[5] A. Masfer and S. Dardouri, "An explainable meta-learning framework for adaptive model selection in short-term load forecasting," Electronics, vol. 15, no. 10, art. 2060, 2026, doi: 10.3390/electronics15102060.
[6] E. Aguilar Madrid, "Short-term electricity load forecasting (Panama case study)," Mendeley Data, V1, 2021, doi: 10.17632/byx7sztj59.1.
[7] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD, 2016.
[8] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Adv. Neural Inf. Process. Syst. 30, 2017.
[9] T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," in Proc. 25th ACM SIGKDD, 2019.
