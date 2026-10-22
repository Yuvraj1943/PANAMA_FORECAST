"""Data loading helpers for the Panama load-forecasting project."""
import pandas as pd

FEATURES = ['week_X-2', 'week_X-3', 'week_X-4', 'MA_X-4', 'dayOfWeek', 'weekend',
            'holiday', 'Holiday_ID', 'hourOfDay', 'T2M_toc']
TARGET = 'DEMAND'


def load_cnd_forecast(path='data/weekly_pre-dispatch_forecast.csv'):
    """CND's own official weekly pre-dispatch forecast (the benchmark to beat)."""
    cnd = pd.read_csv(path)
    cnd['datetime'] = pd.to_datetime(cnd['datetime'], format='%m/%d/%Y %H:%M')
    return cnd.rename(columns={'load_forecast': 'CND'})


def load_folds(train_path='data/train_dataframes.xlsx', test_path='data/test_dataframes.xlsx'):
    """Return a list of (fold_name, train_df, test_df): 14 walk-forward weekly folds."""
    xtr, xte = pd.ExcelFile(train_path), pd.ExcelFile(test_path)
    folds = []
    for name in xtr.sheet_names:
        tr, te = xtr.parse(name), xte.parse(name)
        tr['datetime'] = pd.to_datetime(tr['datetime'])
        te['datetime'] = pd.to_datetime(te['datetime'])
        folds.append((name, tr, te))
    return folds
