"""
Non-linear forecasting models.

Implements a shallow LSTM return forecaster using PyTorch (with a
scikit-learn MLP fallback when PyTorch is not available).
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── PyTorch availability check ────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.info("PyTorch not available; LSTMForecaster will use MLP fallback.")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch LSTM module
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        """Single-layer LSTM followed by a linear output head."""

        def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                     dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head    = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            out    = self.dropout(out[:, -1, :])   # last time-step
            return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTMForecaster
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster:
    """
    Shallow LSTM return forecaster.

    Builds sequences of length *seq_len* from the feature matrix and trains a
    one-layer LSTM to predict the next-step equal-weighted portfolio return.

    Falls back to sklearn MLPRegressor when PyTorch is unavailable.

    Parameters
    ----------
    seq_len : int
        Number of look-back time steps fed into the LSTM per prediction.
    hidden_size : int
        Number of LSTM hidden units.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Adam learning rate.
    patience : int
        Early-stopping patience (epochs without validation improvement).
    """

    def __init__(
        self,
        seq_len: int = 21,
        hidden_size: int = 32,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 10,
    ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.patience    = patience

        self._scaler   = StandardScaler()
        self._model    = None
        self._is_fitted = False
        self._use_torch = _TORCH_AVAILABLE

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMForecaster":
        """
        Fit the LSTM on training data.

        Parameters
        ----------
        X : ndarray of shape (T, n_features)
            Feature matrix (already constructed with lags externally).
        y : ndarray of shape (T,)
            Target returns.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y  = X[valid], y[valid]

        X_s = self._scaler.fit_transform(X)

        if self._use_torch:
            self._fit_torch(X_s, y)
        else:
            self._fit_mlp(X_s, y)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate return predictions.

        Parameters
        ----------
        X : ndarray of shape (T, n_features)

        Returns
        -------
        ndarray of shape (T,)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X   = np.asarray(X, dtype=float)
        X_s = self._scaler.transform(X)

        if self._use_torch:
            return self._predict_torch(X_s)
        else:
            return self._model.predict(X_s)

    # ------------------------------------------------------------------
    # Internal PyTorch helpers
    # ------------------------------------------------------------------

    def _make_sequences(self, X: np.ndarray, y: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (seq_len, n_features) windows from a flat feature matrix."""
        T = len(X)
        seqs, targets = [], []
        for t in range(self.seq_len, T):
            seqs.append(X[t - self.seq_len: t])
            targets.append(y[t])
        return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _fit_torch(self, X_s: np.ndarray, y: np.ndarray) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seqs, tgts = self._make_sequences(X_s, y)

        # 80 / 20 chronological split for early stopping
        n_val   = max(1, int(len(tgts) * 0.2))
        n_train = len(tgts) - n_val

        X_tr = torch.tensor(seqs[:n_train]).to(device)
        y_tr = torch.tensor(tgts[:n_train]).to(device)
        X_val = torch.tensor(seqs[n_train:]).to(device)
        y_val = torch.tensor(tgts[n_train:]).to(device)

        ds     = TensorDataset(X_tr, y_tr)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        net = _LSTMNet(X_s.shape[1], self.hidden_size).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val, best_state, wait = float("inf"), None, 0
        net.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

            with torch.no_grad():
                net.eval()
                val_loss = criterion(net(X_val), y_val).item()
                net.train()

            if val_loss < best_val - 1e-7:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("LSTM early stop at epoch %d (val_loss=%.6f)", epoch, best_val)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._model  = net
        self._device = device
        logger.info("LSTMForecaster (torch) fitted: %d sequences, hidden=%d",
                    n_train, self.hidden_size)

    def _predict_torch(self, X_s: np.ndarray) -> np.ndarray:
        seqs, _ = self._make_sequences(X_s, np.zeros(len(X_s)))
        if len(seqs) == 0:
            return np.full(len(X_s), np.nan)
        t_seqs = torch.tensor(seqs).to(self._device)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(t_seqs).cpu().numpy()
        # Pad the first seq_len predictions with NaN (no look-back available)
        out = np.full(len(X_s), np.nan)
        out[self.seq_len:] = preds
        return out

    # ------------------------------------------------------------------
    # MLP fallback
    # ------------------------------------------------------------------

    def _fit_mlp(self, X_s: np.ndarray, y: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor
        self._model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            n_iter_no_change=10,
        )
        self._model.fit(X_s, y)
        logger.info("LSTMForecaster (MLP fallback) fitted on %d samples.", len(y))


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward LSTM evaluation
# ─────────────────────────────────────────────────────────────────────────────

def rolling_lstm_forecast(
    factor_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    target: pd.Series,
    window: int = 252,
    seq_len: int = 21,
    step: int = 21,
    hidden_size: int = 32,
    epochs: int = 30,
    lags: Optional[list] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Walk-forward LSTM forecaster using lagged factor + macro features.

    Refits the LSTM every *step* days on a rolling *window* of history.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        Latent factor return series (T × K).
    macro_df : pd.DataFrame
        Macro variable series (T × M).
    target : pd.Series
        Equal-weighted portfolio return to forecast.
    window : int
        Training window in days.
    seq_len : int
        LSTM input sequence length.
    step : int
        Days between model refits (for speed).
    hidden_size : int
        LSTM hidden units.
    epochs : int
        Max training epochs per refit.
    lags : list of int, optional
        Feature lags. Defaults to [1, 5].

    Returns
    -------
    predictions : pd.Series
    actuals : pd.Series
    """
    from src.regression_models import build_feature_matrix

    if lags is None:
        lags = [1, 5]

    feature_matrix = build_feature_matrix(factor_returns, macro_df, lags=lags)
    common = feature_matrix.index.intersection(target.index)
    X_all  = feature_matrix.loc[common].values.astype(float)
    y_all  = target.loc[common].values.astype(float)
    dates  = common

    n       = len(dates)
    min_obs = int(window * 0.8)
    preds   = {}
    current_model = None

    logger.info("Rolling LSTM: %d dates, window=%d, seq_len=%d, step=%d", n, window, seq_len, step)

    for i in range(window, n):
        refit = (i - window) % step == 0

        if refit:
            X_win = X_all[i - window: i]
            y_win = y_all[i - window: i]
            valid = np.isfinite(X_win).all(axis=1) & np.isfinite(y_win)
            if valid.sum() < min_obs:
                continue

            current_model = LSTMForecaster(
                seq_len=seq_len, hidden_size=hidden_size,
                epochs=epochs, patience=7,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_model.fit(X_win[valid], y_win[valid])

        if current_model is None:
            continue

        # Predict next step using last seq_len observations before i
        x_hist = X_all[max(0, i - seq_len): i]
        if len(x_hist) < seq_len:
            continue

        x_scaled = current_model._scaler.transform(x_hist)
        if current_model._use_torch:
            import torch
            t_seq = torch.tensor(x_scaled[np.newaxis], dtype=torch.float32).to(
                current_model._device
            )
            current_model._model.eval()
            with torch.no_grad():
                pred = current_model._model(t_seq).cpu().item()
        else:
            pred = current_model._model.predict(x_scaled[[-1]])[0]

        preds[dates[i]] = pred

    predictions = pd.Series(preds, name="lstm_predicted")
    predictions.index = pd.to_datetime(predictions.index)
    actuals = target.loc[predictions.index].rename("actual")

    logger.info("Rolling LSTM done: %d predictions.", len(predictions))
    return predictions, actuals
