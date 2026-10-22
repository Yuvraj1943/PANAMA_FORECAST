"""QA test-suite: data integrity, leakage checks, metric correctness, model sanity, reproducibility.
Run from the project root:  python -m pytest -q"""
import json, joblib, numpy as np, pandas as pd, pytest
from xgboost import XGBRegressor
from src.data import FEATURES, TARGET, load_folds, load_cnd_forecast
from src.metrics import mape, mae, rmse

FOLDS = load_folds()
CND = load_cnd_forecast()
PRED = pd.read_csv('results/predictions_final.csv', parse_dates=['datetime'])


# ---- metric correctness (hand-computed expected values) ----
def test_metrics_known_values():
    y, p = [100, 200], [110, 180]
    assert mape(y, p) == pytest.approx(10.0)
    assert mae(y, p) == pytest.approx(15.0)
    assert rmse(y, p) == pytest.approx(np.sqrt((100 + 400) / 2))


# ---- data integrity ----
def test_fourteen_folds():
    assert len(FOLDS) == 14


@pytest.mark.parametrize('name,tr,te', FOLDS)
def test_fold_integrity(name, tr, te):
    assert len(te) == 168, 'each test week must be 168 hours'
    assert not tr[FEATURES + [TARGET]].isna().any().any(), 'no missing values in training data'
    assert not te[FEATURES].isna().any().any(), 'no missing values in test features'
    assert (te[TARGET] > 0).all()


@pytest.mark.parametrize('name,tr,te', FOLDS)
def test_no_time_leakage(name, tr, te):
    gap_h = (te['datetime'].min() - tr['datetime'].max()).total_seconds() / 3600
    assert gap_h >= 72, f'{name}: need >= 72h gap between train end and test start, got {gap_h}h'
    assert tr['datetime'].max() < te['datetime'].min()


def test_target_not_in_features():
    assert TARGET not in FEATURES


def test_cnd_benchmark_covers_every_test_hour():
    for name, tr, te in FOLDS:
        assert te.merge(CND, on='datetime')['CND'].notna().sum() == 168


# ---- model / results sanity ----
def test_results_match_summary_file():
    s = json.load(open('results/summary.json'))
    assert s['overall']['CND (official)']['MAPE'] == pytest.approx(mape(PRED[TARGET], PRED['CND']), abs=1e-2)
    assert s['overall']['XGBoost (hand-set)']['MAPE'] == pytest.approx(mape(PRED[TARGET], PRED['XGBoost']), abs=1e-2)


def test_xgboost_beats_naive_persistence_and_cnd():
    assert mape(PRED[TARGET], PRED['XGBoost']) < mape(PRED[TARGET], PRED['Persistence(wk-2)'])
    assert mape(PRED[TARGET], PRED['XGBoost']) < mape(PRED[TARGET], PRED['CND'])


def test_predictions_are_plausible():
    for col in ['XGBoost', 'XGBoost_tuned', 'LSTM', 'LinearRegression']:
        assert PRED[col].notna().all()
        assert PRED[col].between(300, 2500).all(), f'{col} outside plausible MWh range'


def test_saved_model_predicts():
    model = joblib.load('results/xgb_Week_24_Jun_2020.joblib')
    te = FOLDS[-1][2]
    out = model.predict(te[FEATURES])
    assert out.shape == (168,) and np.isfinite(out).all()


def test_training_is_reproducible():
    _, tr, te = FOLDS[0]
    kw = dict(n_estimators=50, max_depth=4, random_state=42, n_jobs=1)
    a = XGBRegressor(**kw).fit(tr[FEATURES], tr[TARGET]).predict(te[FEATURES])
    b = XGBRegressor(**kw).fit(tr[FEATURES], tr[TARGET]).predict(te[FEATURES])
    assert np.allclose(a, b)


# ---- reproduction against the source paper (Aguilar Madrid & Antonio, Information 2021, 12(2), 50) ----
PAPER_CND_WEEKLY = [3.90, 3.10, 6.08, 5.55, 4.16, 4.68, 5.04, 6.65, 5.38, 4.06, 3.79, 2.93, 9.06, 4.96]  # Table 5, Pre-Disp. row
PAPER_XGB_AVG = 3.74                                                                                      # reported average MAPE of XGBoost


def _weekly_mape(col):
    return [mape(g[TARGET], g[col]) for _, g in PRED.groupby('fold', sort=False)]


def test_cnd_weekly_errors_match_published_table():
    ours = _weekly_mape('CND')
    assert len(ours) == 14
    assert max(abs(a - b) for a, b in zip(ours, PAPER_CND_WEEKLY)) < 0.01, 'CND weekly errors must match the paper\'s Table 5'


def test_published_settings_reproduce_reported_average():
    ours = float(np.mean(_weekly_mape('XGBoost_paper')))
    assert abs(ours - PAPER_XGB_AVG) < 0.15, f'reproduction {ours:.2f} vs published {PAPER_XGB_AVG}'


def test_published_settings_beat_cnd_every_week():
    x, c = _weekly_mape('XGBoost_paper'), _weekly_mape('CND')
    assert all(a < b for a, b in zip(x, c)), 'the paper reports XGBoost improving on CND in all 14 test weeks'
