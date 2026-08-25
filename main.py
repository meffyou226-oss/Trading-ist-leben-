#!/usr/bin/env python3
"""
main.py
=======
Main entry point for the XAUUSD M5 trading pipeline.

This script orchestrates the full pipeline:
1. Load and validate data
2. Engineer features (with lookahead verification)
3. Compute triple-barrier labels (with optimal barrier search)
4. Train XGBoost models (LONG + SHORT) with walk-forward validation
5. Generate evaluation reports and plots
6. Save trained models

Usage:
    python main.py
    python main.py --skip-barrier-search
    python main.py --horizon 10 --n-splits 5
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import (
    LABEL_CFG, FEATURE_CFG, MODEL_CFG, WF_CFG, EVAL_CFG,
    MODEL_DIR, REPORT_DIR,
)
from pipeline.data_loader import (
    load_data, validate_data, detect_gaps, get_data_summary, add_time_features,
)
from pipeline.features import (
    compute_features, get_feature_columns, verify_no_lookahead,
    get_feature_documentation,
)
from pipeline.labeling import compute_all_labels, find_optimal_barriers
from pipeline.model import walk_forward_train, save_model, get_feature_importance
from pipeline.evaluation import (
    generate_wf_report, plot_feature_importance,
    plot_probability_calibration, simulate_trades,
)

# ─── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="XAUUSD M5 Trading Pipeline")
    parser.add_argument("--skip-barrier-search", action="store_true",
                        help="Skip TP/SL barrier grid search")
    parser.add_argument("--tp-mult", type=float, default=1.5,
                        help="TP multiplier (used if skip-barrier-search)")
    parser.add_argument("--sl-mult", type=float, default=1.0,
                        help="SL multiplier (used if skip-barrier-search)")
    parser.add_argument("--horizon", type=int, default=LABEL_CFG.horizon,
                        help="Triple-barrier horizon in bars")
    parser.add_argument("--n-splits", type=int, default=WF_CFG.n_splits,
                        help="Number of walk-forward splits")
    parser.add_argument("--prob-threshold", type=float, default=EVAL_CFG.probability_threshold,
                        help="Minimum probability for trade signals")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("  XAUUSD M5 TRADING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    # ── Step 1: Load Data ────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 1: Loading data...")
    data = load_data()
    data = add_time_features(data)

    # Print data summary
    summary = get_data_summary(data)
    logger.info("\n" + summary)

    # Save data quality report
    report = validate_data(data)
    gaps = detect_gaps(data)

    os.makedirs(REPORT_DIR, exist_ok=True)
    gaps.to_csv(os.path.join(REPORT_DIR, "data_gaps.csv"), index=False)
    logger.info(f"Data gaps saved to {REPORT_DIR}/data_gaps.csv")

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    logger.info("")
    logger.info("STEP 2: Engineering features...")
    data = compute_features(data)

    feature_cols = get_feature_columns(data)
    logger.info(f"Total features: {len(feature_cols)}")

    # Lookahead verification
    logger.info("Verifying no lookahead leakage...")
    lookahead_results = verify_no_lookahead(data, feature_cols)
    n_pass = sum(lookahead_results.values())
    n_fail = len(lookahead_results) - n_pass
    logger.info(f"Lookahead check: {n_pass} passed, {n_fail} failed")

    if n_fail > 0:
        failed = [k for k, v in lookahead_results.items() if not v]
        logger.warning(f"Features flagged: {failed}")

    # Save feature documentation
    feat_doc = get_feature_documentation()
    feat_doc.to_csv(os.path.join(REPORT_DIR, "feature_documentation.csv"), index=False)

    # ── Step 3: Triple-Barrier Labels ────────────────────────────────────
    logger.info("")
    logger.info("STEP 3: Computing triple-barrier labels...")

    # Get ATR for barrier computation
    atr = data["atr"]

    if not args.skip_barrier_search:
        logger.info("Searching for optimal TP/SL barriers...")
        long_barrier_results = find_optimal_barriers(
            data, atr, args.horizon, "long"
        )
        short_barrier_results = find_optimal_barriers(
            data, atr, args.horizon, "short"
        )
        tp_mult = long_barrier_results["best"]["tp_multiplier"]
        sl_mult = long_barrier_results["best"]["sl_multiplier"]
        logger.info(f"Optimal barriers: TP={tp_mult}x ATR, SL={sl_mult}x ATR")

        # Save barrier search results
        long_barrier_results["all_results"].to_csv(
            os.path.join(REPORT_DIR, "barrier_search_long.csv"), index=False)
        short_barrier_results["all_results"].to_csv(
            os.path.join(REPORT_DIR, "barrier_search_short.csv"), index=False)
    else:
        tp_mult = args.tp_mult
        sl_mult = args.sl_mult
        logger.info(f"Using fixed barriers: TP={tp_mult}x ATR, SL={sl_mult}x ATR")

    # Compute labels
    labels = compute_all_labels(data, atr, tp_mult, sl_mult, args.horizon)

    # Add labels to data
    data["label_long"] = labels["label_long"]
    data["label_short"] = labels["label_short"]

    # ── Step 4: Prepare Feature Matrix ───────────────────────────────────
    logger.info("")
    logger.info("STEP 4: Preparing feature matrix...")

    # Remove rows with NaN in features or labels
    # (NaN arises from indicator warmup periods)
    valid_mask = (
        data[feature_cols].notna().all(axis=1) &
        (data["label_long"] >= 0) &
        (data["label_short"] >= 0)
    )

    X = data.loc[valid_mask, feature_cols].copy()
    y_long = data.loc[valid_mask, "label_long"].copy()
    y_short = data.loc[valid_mask, "label_short"].copy()

    logger.info(f"Valid samples: {len(X):,} (excluded {len(data) - len(X):,})")
    logger.info(f"Feature matrix shape: {X.shape}")

    # Save feature matrix info
    feature_info = pd.DataFrame({
        "feature": feature_cols,
        "mean": X.mean().values,
        "std": X.std().values,
        "min": X.min().values,
        "max": X.max().values,
    })
    feature_info.to_csv(os.path.join(REPORT_DIR, "feature_statistics.csv"), index=False)

    # ── Step 5: Walk-Forward Training ────────────────────────────────────
    logger.info("")
    logger.info("STEP 5: Walk-forward training...")

    # Update config with CLI args
    WF_CFG.n_splits = args.n_splits

    # Train LONG model
    logger.info("")
    logger.info("Training LONG model...")
    long_results = walk_forward_train(X, y_long, direction="long")

    # Train SHORT model
    logger.info("")
    logger.info("Training SHORT model...")
    short_results = walk_forward_train(X, y_short, direction="short")

    # ── Step 6: Evaluation & Reporting ───────────────────────────────────
    logger.info("")
    logger.info("STEP 6: Generating evaluation reports...")

    # Walk-forward report
    report_text = generate_wf_report(long_results, short_results)
    print("\n" + report_text)

    # Feature importance (from last fold model)
    if long_results["models"]:
        long_imp = get_feature_importance(long_results["models"][-1])
        long_imp.to_csv(os.path.join(REPORT_DIR, "feature_importance_long.csv"), index=False)
        plot_feature_importance(
            long_imp, title="LONG Model - Feature Importance",
            output_path=os.path.join(REPORT_DIR, "feature_importance_long.png"),
        )

    if short_results["models"]:
        short_imp = get_feature_importance(short_results["models"][-1])
        short_imp.to_csv(os.path.join(REPORT_DIR, "feature_importance_short.csv"), index=False)
        plot_feature_importance(
            short_imp, title="SHORT Model - Feature Importance",
            output_path=os.path.join(REPORT_DIR, "feature_importance_short.png"),
        )

    # Calibration plots
    if long_results["predictions"]:
        last_pred = long_results["predictions"][-1]
        plot_probability_calibration(
            last_pred["y_true"].values,
            last_pred["y_pred_proba"],
            title="LONG Model - Probability Calibration",
            output_path=os.path.join(REPORT_DIR, "calibration_long.png"),
        )

    if short_results["predictions"]:
        last_pred = short_results["predictions"][-1]
        plot_probability_calibration(
            last_pred["y_true"].values,
            last_pred["y_pred_proba"],
            title="SHORT Model - Probability Calibration",
            output_path=os.path.join(REPORT_DIR, "calibration_short.png"),
        )

    # ── Step 7: Save Models ──────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 7: Saving models...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if long_results["models"]:
        save_model(long_results["models"][-1], "xgboost_long_final")
    if short_results["models"]:
        save_model(short_results["models"][-1], "xgboost_short_final")

    # Save feature names for production use
    import json
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save pipeline config
    config_record = {
        "tp_multiplier": tp_mult,
        "sl_multiplier": sl_mult,
        "horizon": args.horizon,
        "n_splits": args.n_splits,
        "probability_threshold": args.prob_threshold,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "training_date": datetime.now().isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "pipeline_config.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Elapsed time: {elapsed:.1f} seconds")
    logger.info(f"Models saved to: {MODEL_DIR}")
    logger.info(f"Reports saved to: {REPORT_DIR}")
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Training samples: {len(X):,}")

    if long_results["metrics"]:
        metrics_df = pd.DataFrame(long_results["metrics"])
        logger.info(f"LONG model - Avg F1: {metrics_df['f1'].mean():.4f}")
    if short_results["metrics"]:
        metrics_df = pd.DataFrame(short_results["metrics"])
        logger.info(f"SHORT model - Avg F1: {metrics_df['f1'].mean():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
