#!/usr/bin/env python3
"""
main_v2.py
==========
Improved XAUUSD M5 trading pipeline v2.

Key improvements:
1. Better features (volatility regime, microstructure, interactions)
2. Feature selection (correlation filtering + mutual information)
3. Optimal barrier search
4. Hyperparameter tuning
5. Proper walk-forward with embargo
"""

import os
import sys
import argparse
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import MODEL_DIR, REPORT_DIR
from pipeline.data_loader import load_data, add_time_features, validate_data, detect_gaps, get_data_summary
from pipeline.features_v2 import compute_features_v2, get_feature_columns_v2
from pipeline.labeling_v2 import compute_all_labels_v2, find_optimal_barriers_v2
from pipeline.model_v2 import select_features, train_with_walkforward, hyperparameter_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main_v2")


def main():
    parser = argparse.ArgumentParser(description="XAUUSD M5 Pipeline v2")
    parser.add_argument("--horizon", type=int, default=10, help="Triple-barrier horizon")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward splits")
    parser.add_argument("--no-barrier-search", action="store_true", help="Skip barrier search")
    parser.add_argument("--no-hp-search", action="store_true", help="Skip hyperparameter search")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  XAUUSD M5 TRADING PIPELINE v2")
    logger.info("=" * 60)

    # ── Step 1: Load Data ────────────────────────────────────────────────
    logger.info("\nSTEP 1: Loading data...")
    data = load_data()
    data = add_time_features(data)
    summary = get_data_summary(data)
    logger.info("\n" + summary)

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    logger.info("\nSTEP 2: Engineering features (v2)...")
    data = compute_features_v2(data)
    feature_cols = get_feature_columns_v2(data)
    logger.info(f"Raw features: {len(feature_cols)}")

    # ── Step 3: Optimal Barriers ─────────────────────────────────────────
    logger.info("\nSTEP 3: Finding optimal barriers...")

    prev_close = data["close"].shift(1)
    tr = pd.concat([
        (data["high"] - data["low"]).abs(),
        (data["high"] - prev_close).abs(),
        (data["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    if not args.no_barrier_search:
        long_barriers = find_optimal_barriers_v2(data, atr, args.horizon, "long")
        short_barriers = find_optimal_barriers_v2(data, atr, args.horizon, "short")

        tp_mult = long_barriers["best"]["tp_mult"]
        sl_mult = long_barriers["best"]["sl_mult"]

        # Save barrier search results
        os.makedirs(REPORT_DIR, exist_ok=True)
        long_barriers["results"].to_csv(os.path.join(REPORT_DIR, "barrier_search_v2_long.csv"), index=False)
        short_barriers["results"].to_csv(os.path.join(REPORT_DIR, "barrier_search_v2_short.csv"), index=False)
    else:
        tp_mult, sl_mult = 1.5, 1.0

    logger.info(f"Using barriers: TP={tp_mult}x ATR, SL={sl_mult}x ATR")

    # ── Step 4: Compute Labels ───────────────────────────────────────────
    logger.info("\nSTEP 4: Computing labels...")
    labels = compute_all_labels_v2(data, atr, tp_mult, sl_mult, args.horizon)
    data["label_long"] = labels["label_long"]
    data["label_short"] = labels["label_short"]

    # ── Step 5: Prepare Feature Matrix ───────────────────────────────────
    logger.info("\nSTEP 5: Preparing feature matrix...")

    valid_mask = (
        data[feature_cols].notna().all(axis=1) &
        (data["label_long"] >= 0) &
        (data["label_short"] >= 0)
    )

    X = data.loc[valid_mask, feature_cols].copy()
    y_long = data.loc[valid_mask, "label_long"].copy()
    y_short = data.loc[valid_mask, "label_short"].copy()

    logger.info(f"Valid samples: {len(X):,}")
    logger.info(f"Features before selection: {len(feature_cols)}")

    # ── Step 6: Feature Selection ────────────────────────────────────────
    logger.info("\nSTEP 6: Feature selection...")

    # Use LONG labels for feature selection (symmetric for both)
    selected_features = select_features(X, y_long, max_correlation=0.85, max_features=25)
    X_selected = X[selected_features]

    logger.info(f"Selected {len(selected_features)} features:")
    for f in selected_features:
        logger.info(f"  - {f}")

    # ── Step 7: Hyperparameter Search ────────────────────────────────────
    logger.info("\nSTEP 7: Hyperparameter search...")

    if not args.no_hp_search:
        hp_results = hyperparameter_search(X_selected, y_long, n_trials=15)
        best_params = hp_results["best_params"]
    else:
        best_params = {}
        logger.info("Using default hyperparameters")

    # ── Step 8: Walk-Forward Training ────────────────────────────────────
    logger.info("\nSTEP 8: Walk-forward training...")

    logger.info("\nTraining LONG model...")
    long_results = train_with_walkforward(
        X_selected, y_long, direction="long", n_splits=args.n_splits
    )

    logger.info("\nTraining SHORT model...")
    short_results = train_with_walkforward(
        X_selected, y_short, direction="short", n_splits=args.n_splits
    )

    # ── Step 9: Report Results ───────────────────────────────────────────
    logger.info("\nSTEP 9: Results summary...")

    print("\n" + "=" * 60)
    print("  WALK-FORWARD RESULTS v2")
    print("=" * 60)

    for name, results in [("LONG", long_results), ("SHORT", short_results)]:
        if results["metrics"]:
            metrics_df = pd.DataFrame(results["metrics"])
            print(f"\n{name} MODEL:")
            print(f"  Folds: {len(metrics_df)}")
            print(f"  AUC:   {metrics_df['auc'].mean():.4f} ± {metrics_df['auc'].std():.4f}")
            print(f"  F1:    {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}")
            print(f"  Prec:  {metrics_df['precision'].mean():.4f}")
            print(f"  Rec:   {metrics_df['recall'].mean():.4f}")

            print(f"\n  Per-fold AUC:")
            for m in results["metrics"]:
                print(f"    Fold {m['fold']}: AUC={m.get('auc', 'N/A'):.4f}, F1={m['f1']:.4f}")

    # ── Step 10: Save Models ─────────────────────────────────────────────
    logger.info("\nSTEP 10: Saving models...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if long_results["models"]:
        long_results["models"][-1].save_model(os.path.join(MODEL_DIR, "xgboost_long_v2.json"))
    if short_results["models"]:
        short_results["models"][-1].save_model(os.path.join(MODEL_DIR, "xgboost_short_v2.json"))

    # Save feature names and config
    with open(os.path.join(MODEL_DIR, "feature_names_v2.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    config = {
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "horizon": args.horizon,
        "n_splits": args.n_splits,
        "n_features": len(selected_features),
        "features": selected_features,
        "training_date": datetime.now().isoformat(),
        "hyperparameters": best_params,
    }
    with open(os.path.join(MODEL_DIR, "pipeline_config_v2.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    elapsed = time.time() - start_time
    logger.info(f"\nDone in {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
