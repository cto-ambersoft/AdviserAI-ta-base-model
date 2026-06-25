"""
Evaluation utilities: economic (return-based) metrics, naive baselines, and a
probability-calibration metric.

Class mapping: 0=SELL, 1=HOLD, 2=BUY.

Caveat on returns: labels are forward returns over `horizon` bars, so adjacent
bars' returns overlap. `strategy_total_return` therefore sums overlapping
windows and overstates a tradable PnL — but the strategy and the buy-and-hold
baseline use the *same* convention, so comparing their per-bar means is fair.
For an isolated, deflated backtest you would sample non-overlapping bars.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

SELL, HOLD, BUY = 0, 1, 2


def _positions(y_pred: np.ndarray) -> np.ndarray:
    # BUY -> +1 (long), SELL -> -1 (short), HOLD -> 0 (flat)
    return np.where(y_pred == BUY, 1.0, np.where(y_pred == SELL, -1.0, 0.0))


def economic_metrics(y_pred, fwd_return) -> dict[str, float]:
    """Return-based metrics of acting on the predicted signals."""
    y = np.asarray(y_pred).astype(int)
    r = np.asarray(fwd_return, dtype=float)
    pos = _positions(y)
    pnl = pos * r
    traded = pos != 0.0
    n_trades = int(traded.sum())

    return {
        "strategy_mean_return_per_bar": float(pnl.mean()) if pnl.size else 0.0,
        "strategy_total_return": float(pnl.sum()),
        "mean_return_per_trade": float(pnl[traded].mean()) if n_trades else 0.0,
        "n_trades": n_trades,
        "trade_rate": float(traded.mean()) if traded.size else 0.0,
        # fraction of taken trades whose directional bet earned a positive return
        "directional_accuracy": float((pnl[traded] > 0).mean()) if n_trades else 0.0,
    }


def baseline_metrics(y_true, fwd_return) -> dict[str, float]:
    """Naive references to contextualize the model's quality."""
    y = np.asarray(y_true).astype(int)
    r = np.asarray(fwd_return, dtype=float)
    always_hold = np.full(y.shape, HOLD, dtype=int)
    return {
        "buy_and_hold_mean_return_per_bar": float(r.mean()) if r.size else 0.0,
        "always_hold_return": 0.0,
        "always_hold_macro_f1": float(
            f1_score(y, always_hold, labels=[SELL, HOLD, BUY], average="macro", zero_division=0)
        ),
    }


def multiclass_brier_score(prob, y_true, n_classes: int = 3) -> float:
    """
    Multiclass Brier score = mean over samples of sum_k (p_k - onehot_k)^2.

    Equivalent to the (unscaled) multiclass Brier in scikit-learn; lies in [0, 2],
    lower is better, minimized only when predicted probabilities match the truth.
    A strictly-proper scoring rule (mixes calibration and discrimination).
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(y_true).astype(int)
    onehot = np.zeros((y.shape[0], n_classes), dtype=float)
    onehot[np.arange(y.shape[0]), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
