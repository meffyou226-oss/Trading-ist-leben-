"""
config.py
=========
Central configuration for the XAUUSD M5 trading pipeline.
All hyperparameters, paths, and settings are defined here for
full reproducibility.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "..")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

# ─── Data ────────────────────────────────────────────────────────────────────
INSTRUMENT = "XAUUSD"
TIMEFRAME = "M5"
DATA_PATTERN = "XAUUSD_M5_*.csv"

# ─── Triple-Barrier Labeling ─────────────────────────────────────────────────
@dataclass
class LabelConfig:
    """Configuration for triple-barrier labeling."""
    atr_period: int = 14              # ATR lookback for barrier width
    horizon: int = 10                 # Max bars to hold (M5)
    tp_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    sl_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    # Conservative: if both barriers hit in same bar → label as loss
    both_barrier_is_loss: bool = True

# ─── Feature Engineering ─────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # EMA periods
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 100, 200])
    # RSI periods
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    # ROC periods
    roc_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    # ATR period
    atr_period: int = 14
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    # Donchian Channel
    donchian_period: int = 20
    # ADX
    adx_period: int = 14
    # Volatility regimes
    vol_short: int = 10
    vol_long: int = 50
    # Rolling windows for returns
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    # Rolling high/low lookback
    hl_periods: List[int] = field(default_factory=lambda: [10, 20, 50])

# ─── Model Training ──────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Configuration for XGBoost training."""
    # XGBoost hyperparameters
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 50
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    # Early stopping
    early_stopping_rounds: int = 30
    # Class imbalance
    use_scale_pos_weight: bool = True
    # Objective
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"

# ─── Walk-Forward Validation ─────────────────────────────────────────────────
@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    n_splits: int = 5                  # Number of walk-forward windows
    train_ratio: float = 0.7           # Ratio of data for training in each window
    embargo_bars: int = 20             # Purge/embargo bars between train and test
    min_train_bars: int = 5000         # Minimum bars required for training

# ─── Evaluation ──────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    probability_threshold: float = 0.55  # Min probability to take a signal
    cost_per_trade: float = 0.0          # Transaction cost in price units

# ─── Instances ───────────────────────────────────────────────────────────────
LABEL_CFG = LabelConfig()
FEATURE_CFG = FeatureConfig()
MODEL_CFG = ModelConfig()
WF_CFG = WalkForwardConfig()
EVAL_CFG = EvalConfig()
