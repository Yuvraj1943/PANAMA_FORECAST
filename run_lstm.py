"""Step 3: LSTM secondary model, same 14 walk-forward folds, same features.
Each prediction uses a window of the previous 24 hourly feature rows (all of which are
known in advance: calendar, 2-4 week demand lags, temperature), so no future demand leaks in."""
import os, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from src.data import FEATURES, TARGET, load_folds
from src.metrics import mape
tf.keras.utils.set_random_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(4)
WIN = 24

def windows(X, y=None, start=0):
    idx = np.arange(max(start, WIN - 1), len(X))
    xs = np.stack([X[i - WIN + 1:i + 1] for i in idx])
    return xs, (y[idx] if y is not None else None), idx

folds = load_folds(); out = []
for name, tr, te in folds:
    tr = tr.iloc[-17520:].reset_index(drop=True)  # last 2 years only (keeps CPU training time manageable)
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
    ymu, ysd = tr[TARGET].mean(), tr[TARGET].std()
    Xtr = ((tr[FEATURES] - mu) / sd).values.astype('float32'); ytr = ((tr[TARGET] - ymu) / ysd).values.astype('float32')
    allX = np.vstack([Xtr, ((te[FEATURES] - mu) / sd).values.astype('float32')])  # features only (no test demand)
    xs, ys, _ = windows(Xtr, ytr)
    model = tf.keras.Sequential([tf.keras.layers.Input((WIN, len(FEATURES))),
                                 tf.keras.layers.LSTM(32), tf.keras.layers.Dropout(0.1),
                                 tf.keras.layers.Dense(24, activation='relu'), tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    model.fit(xs, ys, epochs=8, batch_size=512, validation_split=0.1, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    te_x, _, _ = windows(allX, start=len(Xtr))
    p = model.predict(te_x, verbose=0).ravel() * ysd + ymu
    out.append(pd.DataFrame({'datetime': te['datetime'], 'fold': name, 'LSTM': p}))
    print(name, 'LSTM MAPE', round(mape(te[TARGET], p), 2), flush=True)
L = pd.concat(out, ignore_index=True); L.to_csv('results/lstm_predictions.csv', index=False)
print('done')
