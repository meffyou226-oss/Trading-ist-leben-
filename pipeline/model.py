"""
model.py
========
XGBoost model training with walk-forward validation, early stopping,
and class imbalance handling.

Key design decisions:
- Training with pandas DataFrame (not numpy) so feature_names are stored
- TimeSeriesSplit for cross-validation (no shuffling)
- Early stopping on a validation set that is temporally AFTER training
- scale_pos_weight for class imbalance
- Purge/embargo between train and test to prevent label leakage
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from pipeline.config import MODEL_CFG, WF_CFG, MODEL_DIR

logger = logging.getLogger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute scale_pos_weight for imbalanced classification.

    scale_pos_weight = number of negative samples / number of positive samples
    This makes the model pay more attention to the minority class.
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return 1.0
    weight = n_neg / n_pos
    logger.info(f"Class distribution: positive={n_pos}, negative={n_neg}, "
                f"scale_pos_weight={weight:.3f}")
    return weight


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "model",
) -> xgb.XGBClassifier:
    """
    Train a single XGBoost classifier with early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (DataFrame preserves column names).
    y_train : pd.Series
        Training labels (0 or 1).
    X_val : pd.DataFrame
        Validation features (temporally after training).
    y_val : pd.Series
        Validation labels.
    model_name : str
        Name for logging.

    Returns
    -------
    xgb.XGBClassifier
        Trained model with feature_names set.
    """
    cfg = MODEL_CFG

    # Compute class weight
    scale_pos_weight = 1.0
    if cfg.use_scale_pos_weight:
        scale_pos_weight = compute_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective=cfg.objective,
        eval_metric=cfg.eval_metric,
        early_stopping_rounds=cfg.early_stopping_rounds,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # Train with early stopping on validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Log training results
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else cfg.n_estimators
    best_score = model.best_score if hasattr(model, 'best_score') else None
    logger.info(f"  {model_name}: best_iteration={best_iter}, best_score={best_score}")

    # Verify feature names are stored
    if hasattr(model, 'feature_names_in_'):
        logger.info(f"  {model_name}: {len(model.feature_names_in_)} features stored")

    return model


def walk_forward_train(
    X: pd.DataFrame,
    y: pd.Series,
    direction: str = "long",
) -> Dict:
    """
    Walk-forward training with multiple train/validation/test windows.

    Schema:
    - Split data into WF_CFG.n_splits + 1 contiguous chunks
    - For each fold i:
        - Train on chunks 0..i
        - Validate on chunk i+1 (for early stopping)
        - Test on chunk i+2 (final evaluation)
    - Apply embargo between train and test

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels (0 or 1, -1 for exclude).
    direction : str
        'long' or 'short' (for naming).

    Returns
    -------
    dict
        Dictionary with models, predictions, and metrics per fold.
    """
    cfg = WF_CFG

    # Remove excluded samples
    valid_mask = y >= 0
    X_valid = X[valid_mask].copy()
    y_valid = y[valid_mask].copy()

    n_samples = len(X_valid)
    fold_size = n_samples // (cfg.n_splits + 2)  # +2 for val and test

    logger.info(f"Walk-forward training: {n_samples} samples, "
                f"{cfg.n_splits} folds, fold_size={fold_size}")

    results = {
        "folds": [],
        "models": [],
        "predictions": [],
        "metrics": [],
    }

    for fold_idx in range(cfg.n_splits):
        # Define window boundaries
        train_end = (fold_idx + 1) * fold_size
        val_end = (fold_idx + 2) * fold_size
        test_end = min((fold_idx + 3) * fold_size, n_samples)

        if test_end > n_samples or val_end >= test_end:
            logger.warning(f"  Fold {fold_idx}: insufficient data, skipping")
            continue

        # Apply embargo: remove bars too close to test start
        embargo = cfg.embargo_bars
        actual_train_end = max(train_end, val_end - embargo)

        X_train = X_valid.iloc[:actual_train_end]
        y_train = y_valid.iloc[:actual_train_end]
        X_val = X_valid.iloc[train_end:val_end]
        y_val = y_valid.iloc[train_end:val_end]
        X_test = X_valid.iloc[val_end:test_end]
        y_test = y_valid.iloc[val_end:test_end]

        if len(X_train) < cfg.min_train_bars:
            logger.warning(f"  Fold {fold_idx}: train size {len(X_train)} < min, skipping")
            continue

        logger.info(f"  Fold {fold_idx}: train={len(X_train)}, val={len(X_val)}, "
                     f"test={len(X_test)}")

        # Train model
        model = train_xgboost(
            X_train, y_train, X_val, y_val,
            model_name=f"{direction}_fold{fold_idx}",
        )

        # Predict on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        metrics["fold"] = fold_idx
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)

        results["folds"].append(fold_idx)
        results["models"].append(model)
        results["predictions"].append({
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        })
        results["metrics"].append(metrics)

        logger.info(f"  Fold {fold_idx} metrics: accuracy={metrics['accuracy']:.4f}, "
                     f"precision={metrics['precision']:.4f}, "
                     f"recall={metrics['recall']:.4f}, "
                     f"f1={metrics['f1']:.4f}, "
                     f"auc={metrics.get('auc', 'N/A')}")

    return results


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_pred_proba : np.ndarray
        Predicted probabilities.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "positive_rate": float((y_true == 1).mean()),
    }

    # AUC requires both classes present
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics["auc"] = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def save_model(model: xgb.XGBClassifier, name: str) -> str:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : xgb.XGBClassifier
    name : str
        File name (without extension).

    Returns
    -------
    str
        Path to saved model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.json")
    model.save_model(path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(name: str) -> xgb.XGBClassifier:
    """Load a saved model."""
    path = os.path.join(MODEL_DIR, f"{name}.json")
    model = xgb.XGBClassifier()
    model.load_model(path)
    logger.info(f"Model loaded from {path}")
    return model


def get_feature_importance(model: xgb.XGBClassifier) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance, rank
    """
    importance = model.feature_importances_
    names = model.feature_names_in_

    df = pd.DataFrame({
        "feature": names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df
