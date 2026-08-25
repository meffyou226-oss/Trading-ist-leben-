"""
labeling.py
===========
Triple-Barrier labeling method for LONG and SHORT setups.

The triple-barrier method defines three barriers:
1. Upper barrier (Take Profit): entry + k1 * ATR
2. Lower barrier (Stop Loss): entry - k2 * ATR
3. Horizontal barrier (Time horizon): max N bars to hold

Label = +1 (positive) if TP is hit before SL within the horizon.
Label = -1 (negative) if SL is hit before TP, or neither is hit by horizon end.

For same-bar hits: conservative approach → label as loss (SL).

Separate symmetric labels for LONG and SHORT models.
"""

import logging
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

from pipeline.config import LABEL_CFG

logger = logging.getLogger(__name__)


def compute_labels(
    data: pd.DataFrame,
    atr: pd.Series,
    tp_multiplier: float,
    sl_multiplier: float,
    horizon: int,
    direction: str = "long",
) -> pd.Series:
    """
    Compute triple-barrier labels for a given direction.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    atr : pd.Series
        ATR series aligned with data index.
    tp_multiplier : float
        TP = entry + tp_multiplier * ATR (for long)
    sl_multiplier : float
        SL = entry - sl_multiplier * ATR (for long)
    horizon : int
        Maximum number of bars to hold.
    direction : str
        'long' or 'short'.

    Returns
    -------
    pd.Series
        Labels: 1 = positive (TP hit first), 0 = negative (SL hit first or timeout).
    """
    n = len(data)
    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values
    atr_vals = atr.values

    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if i + horizon >= n:
            # Not enough future data → exclude from training
            labels[i] = -1  # marker for "exclude"
            continue

        entry = closes[i]
        atr_i = atr_vals[i]

        if atr_i <= 0 or np.isnan(atr_i):
            labels[i] = -1
            continue

        if direction == "long":
            tp_level = entry + tp_multiplier * atr_i
            sl_level = entry - sl_multiplier * atr_i
        else:  # short
            tp_level = entry - tp_multiplier * atr_i
            sl_level = entry + sl_multiplier * atr_i

        # Scan forward within horizon
        hit_tp = False
        hit_sl = False
        hit_bar = -1

        for j in range(1, horizon + 1):
            if i + j >= n:
                break

            if direction == "long":
                if highs[i + j] >= tp_level:
                    hit_tp = True
                    hit_bar = j
                if lows[i + j] <= sl_level:
                    hit_sl = True
                    hit_bar = j
            else:  # short
                if lows[i + j] <= tp_level:
                    hit_tp = True
                    hit_bar = j
                if highs[i + j] >= sl_level:
                    hit_sl = True
                    hit_bar = j

            # If either barrier hit, stop scanning
            if hit_tp or hit_sl:
                break

        # Assign label
        if hit_tp and hit_sl:
            # Both hit in same bar → conservative: treat as loss
            labels[i] = 0
        elif hit_tp:
            labels[i] = 1
        elif hit_sl:
            labels[i] = 0
        else:
            # Neither hit → timeout, treat as loss (no edge)
            labels[i] = 0

    return pd.Series(labels, index=data.index)


def compute_labels_vectorized(
    data: pd.DataFrame,
    atr: pd.Series,
    tp_multiplier: float,
    sl_multiplier: float,
    horizon: int,
    direction: str = "long",
) -> pd.Series:
    """
    Vectorized triple-barrier labeling (much faster than loop).

    Uses rolling max/min over the horizon window to determine barrier hits.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    atr : pd.Series
        ATR series aligned with data index.
    tp_multiplier : float
        TP multiplier on ATR.
    sl_multiplier : float
        SL multiplier on ATR.
    horizon : int
        Maximum number of bars to hold.
    direction : str
        'long' or 'short'.

    Returns
    -------
    pd.Series
        Labels: 1 = positive, 0 = negative, -1 = exclude (insufficient future data).
    """
    n = len(data)
    closes = data["close"].values
    atr_vals = atr.values

    # Pre-compute rolling max/min over the horizon window
    # For each bar i, we need max(high[i+1:i+horizon+1]) and min(low[i+1:i+horizon+1])
    high_series = data["high"]
    low_series = data["low"]

    # Rolling max of future highs (shifted to exclude current bar)
    future_highs = high_series.shift(-1).rolling(window=horizon, min_periods=1).max()
    future_lows = low_series.shift(-1).rolling(window=horizon, min_periods=1).min()

    # For determining which barrier is hit first, we need the first hit bar
    # This requires a loop but we can optimize with numba or early stopping
    # For now, use the loop version for correctness

    return compute_labels(data, atr, tp_multiplier, sl_multiplier, horizon, direction)


def find_optimal_barriers(
    data: pd.DataFrame,
    atr: pd.Series,
    horizon: int = 10,
    direction: str = "long",
    min_positive_rate: float = 0.30,
    max_positive_rate: float = 0.55,
) -> Dict:
    """
    Grid-search for optimal TP/SL multipliers.

    Searches over combinations of tp_multipliers and sl_multipliers
    to find a pair that yields a positive rate within the target range
    (not too easy, not too hard for the model).

    Parameters
    ----------
    data : pd.DataFrame
    atr : pd.Series
    horizon : int
    direction : str
    min_positive_rate : float
    max_positive_rate : float

    Returns
    -------
    dict
        Best parameters and search results.
    """
    cfg = LABEL_CFG
    best_score = -1
    best_params = {}
    results = []

    for tp_mult in cfg.tp_multipliers:
        for sl_mult in cfg.sl_multipliers:
            labels = compute_labels(data, atr, tp_mult, sl_mult, horizon, direction)

            # Exclude -1 labels
            valid = labels[labels >= 0]
            if len(valid) == 0:
                continue

            positive_rate = (valid == 1).mean()

            # Score: prefer positive rate close to 40-45% (balanced but learnable)
            target_rate = 0.40
            score = -abs(positive_rate - target_rate)

            results.append({
                "tp_multiplier": tp_mult,
                "sl_multiplier": sl_mult,
                "positive_rate": positive_rate,
                "n_valid": len(valid),
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_params = {
                    "tp_multiplier": tp_mult,
                    "sl_multiplier": sl_mult,
                    "positive_rate": positive_rate,
                }

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)
    logger.info(f"Best barriers for {direction}: TP={best_params.get('tp_multiplier')}x ATR, "
                f"SL={best_params.get('sl_multiplier')}x ATR, "
                f"positive rate={best_params.get('positive_rate', 0):.3f}")

    return {
        "best": best_params,
        "all_results": results_df,
    }


def compute_all_labels(
    data: pd.DataFrame,
    atr: pd.Series,
    tp_multiplier: float,
    sl_multiplier: float,
    horizon: int,
) -> pd.DataFrame:
    """
    Compute labels for both LONG and SHORT directions.

    Returns
    -------
    pd.DataFrame
        Columns: label_long, label_short
    """
    logger.info("Computing LONG labels...")
    label_long = compute_labels(data, atr, tp_multiplier, sl_multiplier, horizon, "long")

    logger.info("Computing SHORT labels...")
    label_short = compute_labels(data, atr, tp_multiplier, sl_multiplier, horizon, "short")

    labels = pd.DataFrame({
        "label_long": label_long,
        "label_short": label_short,
    }, index=data.index)

    # Report class distribution
    for col in ["label_long", "label_short"]:
        valid = labels[col][labels[col] >= 0]
        n_pos = (valid == 1).sum()
        n_neg = (valid == 0).sum()
        n_excl = (labels[col] == -1).sum()
        logger.info(f"  {col}: positive={n_pos} ({n_pos/len(valid):.3f}), "
                     f"negative={n_neg}, excluded={n_excl}")

    return labels
