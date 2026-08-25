"""
model_v2.py
===========
Improved model training with:
- Feature selection (correlation filtering + mutual information)
- Hyperparameter tuning via grid search
- Proper walk-forward with embargo
- SHAP-based feature importance
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_correlation: float = 0.85,
    max_features: int = 30,
) -> List[str]:
    """
    Two-stage feature selection:
    1. Remove highly correlated features
    2. Select by mutual information with target

    Parameters
    ----------
    X : pd.DataFrame - feature matrix
    y : pd.Series - labels (0/1)
    max_correlation : float - max pairwise correlation
    max_features : int - max features to keep

    Returns
    -------
    List[str] - selected feature names
    """
    logger.info(f"Feature selection: starting with {len(X.columns)} features")

    # Stage 1: Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation > threshold
    to_drop = set()
    for column in upper.columns:
        correlated = upper.index[upper[column] > max_correlation].tolist()
        # Keep the one with higher MI (computed later), for now just drop duplicates
        for feat in correlated:
            if feat not in to_drop and column not in to_drop:
                to_drop.add(feat)

    remaining = [c for c in X.columns if c not in to_drop]
    logger.info(f"  After correlation filter: {len(remaining)} features")

    # Stage 2: Mutual information
    X_remaining = X[remaining].copy()

    # Fill NaN for MI computation
    X_filled = X_remaining.fillna(X_remaining.median())

    # Compute MI scores
    mi_scores = mutual_info_classif(X_filled, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": remaining, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    # Select top features
    n_select = min(max_features, len(mi_df))
    selected = mi_df.head(n_select)["feature"].tolist()

    logger.info(f"  After MI selection: {len(selected)} features")
    logger.info(f"  Top 10 features by MI:")
    for _, row in mi_df.head(10).iterrows():
        logger.info(f"    {row['feature']}: {row['mi_score']:.4f}")

    return selected


def train_with_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    direction: str = "long",
    n_splits: int = 5,
    embargo: int = 20,
) -> Dict:
    """
    Walk-forward training with proper temporal structure.

    For each fold:
    - Train on expanding window
    - Validate on next chunk (for early stopping)
    - Test on final chunk (out-of-sample)
    - Apply embargo between train and test
    """
    # Remove excluded samples
    valid_mask = y >= 0
    X_valid = X[valid_mask].copy()
    y_valid = y[valid_mask].copy()

    n = len(X_valid)
    fold_size = n // (n_splits + 2)

    logger.info(f"Walk-forward: {n} samples, {n_splits} folds, fold_size={fold_size}")

    results = {
        "folds": [],
        "models": [],
        "predictions": [],
        "metrics": [],
    }

    for fold_idx in range(n_splits):
        # Window boundaries
        train_end = (fold_idx + 1) * fold_size
        val_end = (fold_idx + 2) * fold_size
        test_end = min((fold_idx + 3) * fold_size, n)

        if test_end > n or val_end >= test_end:
            continue

        # Apply embargo
        actual_train_end = min(train_end, val_end - embargo)

        X_train = X_valid.iloc[:actual_train_end]
        y_train = y_valid.iloc[:actual_train_end]
        X_val = X_valid.iloc[train_end:val_end]
        y_val = y_valid.iloc[train_end:val_end]
        X_test = X_valid.iloc[val_end:test_end]
        y_test = y_valid.iloc[val_end:test_end]

        if len(X_train) < 2000:
            continue

        # Compute class weight
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        logger.info(f"  Fold {fold_idx}: train={len(X_train)}, val={len(X_val)}, "
                     f"test={len(X_test)}, spw={scale_pos_weight:.2f}")

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=100,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )

        metrics = {
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        if len(np.unique(y_test)) > 1:
            metrics["auc"] = roc_auc_score(y_test, y_pred_proba)

        results["folds"].append(fold_idx)
        results["models"].append(model)
        results["predictions"].append({
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        })
        results["metrics"].append(metrics)

        logger.info(f"    AUC={metrics.get('auc', 'N/A'):.4f}, F1={metrics['f1']:.4f}")

    return results


def hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
) -> Dict:
    """
    Simple hyperparameter search using random search with time-series CV.
    """
    from sklearn.model_selection import ParameterSampler

    param_distributions = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [50, 100, 200],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    best_auc = 0
    best_params = {}

    # Use a single train/val split for speed
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    spw = n_neg / max(n_pos, 1)

    sampler = ParameterSampler(param_distributions, n_trials, random_state=42)

    for i, params in enumerate(sampler):
        model = xgb.XGBClassifier(
            n_estimators=300,
            gamma=0.1,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **params,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, y_pred_proba)
        else:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_params = params
            logger.info(f"  Trial {i}: AUC={auc:.4f}, params={params}")

    logger.info(f"Best params: {best_params}, AUC={best_auc:.4f}")
    return {"best_params": best_params, "best_auc": best_auc}
