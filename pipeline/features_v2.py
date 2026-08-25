"""
features_v2.py
==============
Improved feature engineering for XAUUSD M5 trading.

Key improvements over v1:
- Focus on VOLATILITY REGIME features (strong autocorrelation = predictable)
- Microstructure features (order flow proxy from OHLC)
- Inter-session dynamics
- Fewer but more predictive features
- All features are dimensionless (ratios/percentages)
"""

import logging
from typing import List, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_features_v2(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute improved feature set focused on regime detection
    and microstructure patterns.
    """
    df = data.copy()
    features = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    prev_close = close.shift(1)

    # ── 1. Volatility Features (strongest predictor) ────────────────────
    logger.info("Computing volatility regime features...")

    # True Range and ATR
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_14 = tr.ewm(span=14, adjust=False).mean()
    atr_50 = tr.ewm(span=50, adjust=False).mean()
    atr_200 = tr.ewm(span=200, adjust=False).mean()

    features["atr_pct"] = atr_14 / close * 100

    # ATR momentum (acceleration/deceleration of volatility)
    features["atr_momentum"] = (atr_14 / atr_50 - 1) * 100
    features["atr_regime"] = (atr_14 / atr_200 - 1) * 100

    # ATR percentile rank (where are we in the recent distribution?)
    def _rank_pct(x):
        if len(x) == 0:
            return np.nan
        return (x[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50.0
    features["atr_percentile"] = atr_14.rolling(200, min_periods=50).apply(_rank_pct, raw=True)

    # Volatility of volatility (second-order)
    atr_std = atr_14.rolling(50, min_periods=20).std()
    features["atr_volatility"] = atr_std / atr_14 * 100

    # Realized volatility at different horizons
    ret_1 = close.pct_change()
    features["realized_vol_5"] = ret_1.rolling(5, min_periods=3).std() * 100
    features["realized_vol_20"] = ret_1.rolling(20, min_periods=10).std() * 100
    features["realized_vol_ratio"] = (
        features["realized_vol_5"] / features["realized_vol_20"].replace(0, np.nan)
    )

    # Garman-Klass volatility estimator (more efficient than close-to-close)
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    features["gk_vol_10"] = np.sqrt(gk.rolling(10, min_periods=5).mean()) * 100

    # ── 2. Trend Features (distance-based, dimensionless) ───────────────
    logger.info("Computing trend features...")

    for period in [21, 50, 100, 200]:
        ema = close.ewm(span=period, adjust=False).mean()
        features[f"dist_ema_{period}"] = (close / ema - 1) * 100

    # EMA alignment (all EMAs stacked correctly = strong trend)
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_100 = close.ewm(span=100, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()

    # EMA alignment (all EMAs stacked correctly = strong trend)
    features["ema_alignment"] = (
        (ema_21 > ema_50).astype(float) +
        (ema_50 > ema_100).astype(float) +
        (close > ema_21).astype(float)
    ) / 3 * 100  # 0=bearish, 100=bullish

    # Trend strength (distance between fast and slow EMAs)
    features["trend_strength"] = (ema_21 / ema_200 - 1) * 100

    # ── 3. Microstructure Features (OHLC patterns) ──────────────────────
    logger.info("Computing microstructure features...")

    candle_range = (high - low).replace(0, np.nan)

    # Body ratio (where does close sit in the candle?)
    features["body_ratio"] = (close - low) / candle_range * 100  # 0=weak, 100=strong
    features["body_size"] = (close - open_).abs() / candle_range * 100

    # Upper/lower shadow ratios
    features["upper_shadow"] = (high - close.clip(lower=open_)) / candle_range * 100
    features["lower_shadow"] = (close.clip(upper=open_) - low) / candle_range * 100

    # Close position relative to recent range (Stochastic-like)
    for period in [10, 20]:
        hh = high.rolling(period, min_periods=5).max()
        ll = low.rolling(period, min_periods=5).min()
        rng = (hh - ll).replace(0, np.nan)
        features[f"close_position_{period}"] = (close - ll) / rng * 100

    # Consecutive same-direction bars
    direction = np.sign(close - open_)
    features["consecutive_bars"] = _consecutive_sum(direction)

    # Gap size
    features["gap"] = (open_ / prev_close - 1) * 100

    # ── 4. Momentum Features ────────────────────────────────────────────
    logger.info("Computing momentum features...")

    # RSI with multiple periods
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta).where(delta < 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        features[f"rsi_{period}"] = (100 - 100 / (1 + rs)).fillna(50)

    # RSI divergence (short-term RSI vs long-term RSI)
    features["rsi_divergence"] = features["rsi_7"] - features["rsi_21"]

    # Rate of Change at multiple horizons
    for period in [5, 10, 20]:
        features[f"roc_{period}"] = (close / close.shift(period) - 1) * 100

    # Momentum acceleration
    features["momentum_accel"] = features["roc_5"] - features["roc_10"]

    # ── 5. Bollinger Band Features ──────────────────────────────────────
    logger.info("Computing Bollinger features...")

    bb_mid = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_width = (4 * bb_std / bb_mid) * 100  # Percentage width

    features["bb_position"] = (close - (bb_mid - 2*bb_std)) / (4 * bb_std).replace(0, np.nan) * 100
    features["bb_width"] = bb_width
    features["bb_squeeze"] = bb_width.rolling(50, min_periods=20).apply(_rank_pct, raw=True)

    # ── 6. ADX / Directional ────────────────────────────────────────────
    logger.info("Computing ADX features...")

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=14, adjust=False).mean()

    features["adx"] = adx
    features["di_diff"] = plus_di - minus_di
    features["adx_regime"] = adx.rolling(50, min_periods=20).apply(_rank_pct, raw=True)

    # ── 7. Volume Features ──────────────────────────────────────────────
    logger.info("Computing volume features...")

    vol_sma_20 = volume.rolling(20, min_periods=10).mean()
    vol_sma_50 = volume.rolling(50, min_periods=20).mean()

    features["vol_ratio"] = volume / vol_sma_20.replace(0, np.nan)
    features["vol_trend"] = (volume / vol_sma_50.replace(0, np.nan) - 1) * 100

    # On-Balance-Volume proxy (using close direction)
    obv = (np.sign(close.diff()) * volume).cumsum()
    features["obv_slope"] = (obv / obv.shift(20) - 1) * 100

    # Volume-weighted price position
    features["vol_price_corr"] = (
        (close - close.shift(1)).rolling(20).corr(volume)
    )

    # ── 8. Session/Time Features (cyclical) ─────────────────────────────
    logger.info("Computing session features...")

    hour = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek

    # Cyclical encodings
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Session indicators
    features["session_ny"] = ((hour >= 12) & (hour < 20)).astype(float)
    features["session_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    features["session_asia"] = ((hour >= 23) | (hour < 8)).astype(float)

    # ── 9. Return Distribution Features ─────────────────────────────────
    logger.info("Computing return distribution features...")

    # Skewness and kurtosis of recent returns
    features["ret_skew_20"] = ret_1.rolling(20, min_periods=10).skew()
    features["ret_skew_50"] = ret_1.rolling(50, min_periods=20).skew()
    features["ret_kurt_20"] = ret_1.rolling(20, min_periods=10).kurt()

    # Max drawdown over recent horizon
    rolling_max = close.rolling(20, min_periods=10).max()
    features["drawdown_20"] = (close / rolling_max - 1) * 100

    # ── 10. Cross-Feature Interactions ──────────────────────────────────
    logger.info("Computing interaction features...")

    # Volatility × Trend interaction
    features["vol_trend_interact"] = features["atr_pct"] * abs(features["dist_ema_50"])

    # Momentum × Volume confirmation
    features["mom_vol_confirm"] = features["roc_10"] * features["vol_ratio"]

    # Session × Volatility
    features["session_vol"] = features["atr_pct"] * features["session_ny"]

    logger.info(f"Total features computed: {len(features.columns)}")

    # Merge with original data
    result = pd.concat([df, features], axis=1)
    return result


def _consecutive_sum(direction: pd.Series) -> pd.Series:
    """Sum consecutive same-direction values (signed count)."""
    result = pd.Series(0.0, index=direction.index)
    count = 0
    for i in range(len(direction)):
        if i > 0 and direction.iloc[i] == direction.iloc[i-1]:
            count += 1 if direction.iloc[i] > 0 else -1
        else:
            count = direction.iloc[i]
        result.iloc[i] = count
    return result


def get_feature_columns_v2(data: pd.DataFrame) -> List[str]:
    """Return feature column names (exclude raw OHLCV and time columns)."""
    exclude = {"timestamp", "date", "time", "open", "high", "low", "close",
               "volume", "datetime", "hour", "dayofweek"}
    return [c for c in data.columns if c not in exclude and c in data.columns]
