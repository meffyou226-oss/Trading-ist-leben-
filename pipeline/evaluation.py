"""
evaluation.py
=============
Evaluation and reporting module for the trading pipeline.

Generates comprehensive reports including:
- Walk-forward performance summary
- Feature importance analysis
- Probability calibration analysis
- Equity curve simulation
"""

import os
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import REPORT_DIR, EVAL_CFG

logger = logging.getLogger(__name__)


def generate_wf_report(
    long_results: Dict,
    short_results: Dict,
    output_dir: str = REPORT_DIR,
) -> str:
    """
    Generate a comprehensive walk-forward validation report.

    Parameters
    ----------
    long_results : dict
        Results from walk_forward_train for LONG model.
    short_results : dict
        Results from walk_forward_train for SHORT model.
    output_dir : str
        Directory to save the report.

    Returns
    -------
    str
        Path to the saved report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "walk_forward_report.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  WALK-FORWARD VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # ── LONG Model ───────────────────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("  LONG MODEL")
    lines.append("-" * 70)

    if long_results["metrics"]:
        metrics_df = pd.DataFrame(long_results["metrics"])
        lines.append(f"  Number of folds:    {len(metrics_df)}")
        lines.append(f"  Accuracy:           {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}")
        lines.append(f"  Precision:          {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
        lines.append(f"  Recall:             {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
        lines.append(f"  F1 Score:           {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}")
        auc_vals = metrics_df['auc'].dropna()
        if len(auc_vals) > 0:
            lines.append(f"  AUC-ROC:            {auc_vals.mean():.4f} ± {auc_vals.std():.4f}")
        lines.append(f"  Avg Positive Rate:  {metrics_df['positive_rate'].mean():.4f}")
        lines.append("")

        # Per-fold details
        lines.append("  Per-fold metrics:")
        for m in long_results["metrics"]:
            lines.append(f"    Fold {m['fold']}: acc={m['accuracy']:.4f}, "
                         f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, "
                         f"f1={m['f1']:.4f}, test_size={m['test_size']}")
    else:
        lines.append("  No folds completed.")

    lines.append("")

    # ── SHORT Model ──────────────────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("  SHORT MODEL")
    lines.append("-" * 70)

    if short_results["metrics"]:
        metrics_df = pd.DataFrame(short_results["metrics"])
        lines.append(f"  Number of folds:    {len(metrics_df)}")
        lines.append(f"  Accuracy:           {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}")
        lines.append(f"  Precision:          {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
        lines.append(f"  Recall:             {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
        lines.append(f"  F1 Score:           {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}")
        auc_vals = metrics_df['auc'].dropna()
        if len(auc_vals) > 0:
            lines.append(f"  AUC-ROC:            {auc_vals.mean():.4f} ± {auc_vals.std():.4f}")
        lines.append(f"  Avg Positive Rate:  {metrics_df['positive_rate'].mean():.4f}")
        lines.append("")

        lines.append("  Per-fold metrics:")
        for m in short_results["metrics"]:
            lines.append(f"    Fold {m['fold']}: acc={m['accuracy']:.4f}, "
                         f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, "
                         f"f1={m['f1']:.4f}, test_size={m['test_size']}")
    else:
        lines.append("  No folds completed.")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path}")
    return report_text


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 25,
    output_path: str = None,
) -> str:
    """
    Plot feature importance as a horizontal bar chart.

    Parameters
    ----------
    importance_df : pd.DataFrame
        From get_feature_importance().
    title : str
    top_n : int
        Show top N features.
    output_path : str
        Path to save the plot.

    Returns
    -------
    str
        Path to saved plot.
    """
    os.makedirs(os.path.dirname(output_path) if output_path else REPORT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, "feature_importance.png")

    top = importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    ax.barh(top["feature"], top["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature importance plot saved to {output_path}")
    return output_path


def plot_probability_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Probability Calibration",
    output_path: str = None,
) -> str:
    """
    Plot probability calibration curve (reliability diagram).

    Shows whether predicted probabilities match actual frequencies.
    """
    os.makedirs(os.path.dirname(output_path) if output_path else REPORT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, "calibration.png")

    # Bin predictions
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(y_pred_proba, bin_edges[1:])

    observed_freq = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            observed_freq[i] = y_true[mask].mean()
            bin_counts[i] = mask.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, observed_freq, width=0.08, alpha=0.7, label="Observed")
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Observed Frequency")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Histogram of predictions
    ax2.hist(y_pred_proba, bins=20, alpha=0.7, color="steelblue")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Calibration plot saved to {output_path}")
    return output_path


def plot_equity_curve(
    returns: pd.Series,
    title: str = "Equity Curve",
    output_path: str = None,
) -> str:
    """
    Plot cumulative equity curve from returns series.
    """
    os.makedirs(os.path.dirname(output_path) if output_path else REPORT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, "equity_curve.png")

    equity = (1 + returns).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.values, color="steelblue", linewidth=1)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (multiple)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Equity curve saved to {output_path}")
    return output_path


def simulate_trades(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    probability_threshold: float = EVAL_CFG.probability_threshold,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Simulate trades based on model predictions.

    Parameters
    ----------
    y_true : pd.Series
        True labels (1 = TP hit, 0 = SL hit).
    y_pred_proba : np.ndarray
        Predicted probabilities.
    probability_threshold : float
        Minimum probability to take a trade.
    direction : str
        'long' or 'short'.

    Returns
    -------
    pd.DataFrame
        Trade log with columns: entry_idx, probability, outcome, return
    """
    trades = []
    for i in range(len(y_true)):
        if y_pred_proba[i] >= probability_threshold:
            outcome = 1 if y_true.iloc[i] == 1 else 0
            trades.append({
                "entry_idx": i,
                "probability": y_pred_proba[i],
                "outcome": outcome,
            })

    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0:
        win_rate = trades_df["outcome"].mean()
        logger.info(f"  {direction} trades: {len(trades_df)} signals, "
                     f"win_rate={win_rate:.3f}")

    return trades_df
