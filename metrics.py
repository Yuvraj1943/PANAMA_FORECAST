"""Forecast accuracy metrics (same definitions as used in the report)."""
import numpy as np


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def summary(y, p):
    return {'MAPE': mape(y, p), 'MAE': mae(y, p), 'RMSE': rmse(y, p)}
