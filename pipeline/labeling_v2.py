"""
labeling_v2.py
==============
Improved triple-barrier labeling with:
- Optimal barrier search via grid search
- Meta-labeling support (separate primary and secondary labels)
- Symmetric LONG/SHORT labels
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_labels_v2(
    data: pd.DataFrame,
    atr: pd.Series,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    direction: str = "long",
) -> pd.Series:
    """
    Optimized triple-barrier labeling using vectorized operations.

    Parameters
    ----------
    data : pd.DataFrame with high, low, close
    atr : pd.Series - aligned ATR
    tp_mult : float - TP multiplier
    sl_mult : float - SL multiplier
    horizon : int - max holding bars
    direction : str - 'long' or 'short'

    Returns
    -------
    pd.Series - 1=TP hit first, 0=SL hit first or timeout, -1=exclude
    """
    n = len(data)
    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values
    atr_vals = atr.values

    labels = np.full(n, -1, dtype=np.int8)

    # Process in chunks for memory efficiency
    chunk_size = 5000

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)

        for i in range(chunk_start, chunk_end):
            if i + 2 >= n:  # Need at least 1 future bar
                continue

            atr_i = atr_vals[i]
            if atr_i <= 0 or np.isnan(atr_i):
                continue

            entry = closes[i]
            max_look = min(horizon, n - i - 1)

            if direction == "long":
                tp_level = entry + tp_mult * atr_i
                sl_level = entry - sl_mult * atr_i

                # Scan forward
                for j in range(1, max_look + 1):
                    h = highs[i + j]
                    l = lows[i + j]

                    tp_hit = h >= tp_level
                    sl_hit = l <= sl_level

                    if tp_hit and sl_hit:
                        labels[i] = 0  # Conservative: both = loss
                        break
                    elif tp_hit:
                        labels[i] = 1
                        break
                    elif sl_hit:
                        labels[i] = 0
                        break
                else:
                    labels[i] = 0  # Timeout = loss

            else:  # short
                tp_level = entry - tp_mult * atr_i
                sl_level = entry + sl_mult * atr_i

                for j in range(1, max_look + 1):
                    h = highs[i + j]
                    l = lows[i + j]

                    tp_hit = l <= tp_level
                    sl_hit = h >= sl_level

                    if tp_hit and sl_hit:
                        labels[i] = 0
                        break
                    elif tp_hit:
                        labels[i] = 1
                        break
                    elif sl_hit:
                        labels[i] = 0
                        break
                else:
                    labels[i] = 0

    return pd.Series(labels, index=data.index)


def find_optimal_barriers_v2(
    data: pd.DataFrame,
    atr: pd.Series,
    horizon: int = 10,
    direction: str = "long",
) -> Dict:
    """
    Grid search for optimal TP/SL multipliers.

    Objective: maximize the "learnability" of the labels by finding
    a TP/SL ratio that produces:
    - Positive rate between 35-50% (balanced enough to learn)
    - Good separation between win/loss returns
    """
    tp_range = np.arange(0.8, 3.1, 0.2)
    sl_range = np.arange(0.5, 2.6, 0.2)

    best_score = -np.inf
    best_params = {}
    results = []

    closes = data["close"].values
    n = len(data)

    for tp_mult in tp_range:
        for sl_mult in sl_range:
            # Use a subsample for speed
            step = max(1, n // 10000)
            indices = list(range(0, n - horizon - 1, step))

            pos_count = 0
            neg_count = 0
            win_returns = []
            loss_returns = []

            for i in indices:
                atr_i = atr.iloc[i]
                if atr_i <= 0 or np.isnan(atr_i):
                    continue

                entry = closes[i]
                max_look = min(horizon, n - i - 1)

                if direction == "long":
                    tp_level = entry + tp_mult * atr_i
                    sl_level = entry - sl_mult * atr_i
                else:
                    tp_level = entry - tp_mult * atr_i
                    sl_level = entry + sl_mult * atr_i

                result = 0
                ret = 0
                for j in range(1, max_look + 1):
                    h = data["high"].iloc[i + j]
                    l = data["low"].iloc[i + j]

                    if direction == "long":
                        tp_hit = h >= tp_level
                        sl_hit = l <= sl_level
                    else:
                        tp_hit = l <= tp_level
                        sl_hit = h >= sl_level

                    if tp_hit and sl_hit:
                        result = 0
                        ret = -sl_mult * atr_i / entry * 100
                        break
                    elif tp_hit:
                        result = 1
                        ret = tp_mult * atr_i / entry * 100
                        break
                    elif sl_hit:
                        result = 0
                        ret = -sl_mult * atr_i / entry * 100
                        break
                else:
                    # Timeout - use final return
                    final_ret = (closes[i + max_look] / entry - 1) * 100
                    ret = final_ret
                    result = 1 if final_ret > 0 else 0

                if result == 1:
                    pos_count += 1
                    win_returns.append(ret)
                else:
                    neg_count += 1
                    loss_returns.append(ret)

            total = pos_count + neg_count
            if total < 100:
                continue

            pos_rate = pos_count / total
            avg_win = np.mean(win_returns) if win_returns else 0
            avg_loss = np.mean(loss_returns) if loss_returns else 0

            # Scoring: prefer balanced classes with positive expectancy
            expectancy = (pos_rate * avg_win + (1 - pos_rate) * avg_loss)
            balance_score = -abs(pos_rate - 0.40)  # Prefer ~40% positive
            score = expectancy + balance_score * 0.5

            results.append({
                "tp_mult": round(tp_mult, 1),
                "sl_mult": round(sl_mult, 1),
                "pos_rate": round(pos_rate, 3),
                "avg_win": round(avg_win, 4),
                "avg_loss": round(avg_loss, 4),
                "expectancy": round(expectancy, 4),
                "score": round(score, 4),
            })

            if score > best_score:
                best_score = score
                best_params = {
                    "tp_mult": round(tp_mult, 1),
                    "sl_mult": round(sl_mult, 1),
                    "pos_rate": round(pos_rate, 3),
                    "expectancy": round(expectancy, 4),
                }

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)
    logger.info(f"Optimal barriers for {direction}: TP={best_params['tp_mult']}x, SL={best_params['sl_mult']}x, "
                f"pos_rate={best_params['pos_rate']:.3f}, expectancy={best_params['expectancy']:.4f}")

    return {"best": best_params, "results": results_df}


def compute_all_labels_v2(
    data: pd.DataFrame,
    atr: pd.Series,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
) -> pd.DataFrame:
    """Compute labels for both LONG and SHORT."""
    logger.info("Computing LONG labels...")
    label_long = compute_labels_v2(data, atr, tp_mult, sl_mult, horizon, "long")

    logger.info("Computing SHORT labels...")
    label_short = compute_labels_v2(data, atr, tp_mult, sl_mult, horizon, "short")

    labels = pd.DataFrame({
        "label_long": label_long,
        "label_short": label_short,
    }, index=data.index)

    for col in ["label_long", "label_short"]:
        valid = labels[col][labels[col] >= 0]
        n_pos = (valid == 1).sum()
        n_neg = (valid == 0).sum()
        logger.info(f"  {col}: pos={n_pos} ({n_pos/len(valid):.3f}), neg={n_neg}")

    return labels
