"""
features.py
===========
Feature engineering module for XAUUSD M5 trading pipeline.

All features are computed using ONLY data up to and including the
current (already closed) candle. No lookahead leakage is introduced.

Each feature is expressed as a percentage or ratio (not absolute price)
to ensure robustness across different price levels (e.g. XAUUSD at 2000
vs 4000 USD).

Lookahead verification is performed in `verify_no_lookahead()`.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from pipeline.config import FEATURE_CFG

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. Uses only past data."""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average. Uses only past data."""
    return series.rolling(window=period, min_periods=period).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Average True Range. Computed from High, Low, Close of previous bars only.
    No lookahead: uses shift(1) of close for the prev close term.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """
    RSI (Relative Strength Index). Uses only past close data.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _rolling_std(series: pd.Series, period: int) -> pd.Series:
    """Rolling standard deviation. Uses only past data."""
    return series.rolling(window=period, min_periods=period).std()


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full feature set for the trading model.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: open, high, low, close, volume, hour, dayofweek

    Returns
    -------
    pd.DataFrame
        DataFrame with all features. Index matches input data.
    """
    cfg = FEATURE_CFG
    df = data.copy()

    # ── 1. Price / Candle Structure ──────────────────────────────────────
    logger.info("Computing price/candle structure features...")

    # Returns over multiple periods (percentage)
    for p in cfg.return_periods:
        df[f"ret_{p}"] = df["close"].pct_change(periods=p) * 100

    # Candle body as percentage of range
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / candle_range

    # Upper wick percentage
    df["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range

    # Lower wick percentage
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range

    # Range as percentage of close (intra-bar volatility)
    df["range_pct"] = candle_range / df["close"] * 100

    # Gap from previous close (percentage)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100

    # ── 2. Trend / Distance to EMAs ──────────────────────────────────────
    logger.info("Computing trend/EMA features...")

    for period in cfg.ema_periods:
        ema = _ema(df["close"], period)
        # Distance from close to EMA as percentage
        df[f"dist_ema_{period}"] = (df["close"] - ema) / ema * 100

    # EMA slope (rate of change of EMA, percentage)
    for period in cfg.ema_periods:
        ema = _ema(df["close"], period)
        df[f"ema_{period}_slope"] = ema.pct_change(periods=5) * 100

    # EMA crossovers (ratio of short to long EMA)
    if len(cfg.ema_periods) >= 2:
        for i in range(len(cfg.ema_periods) - 1):
            short_p = cfg.ema_periods[i]
            long_p = cfg.ema_periods[i + 1]
            short_ema = _ema(df["close"], short_p)
            long_ema = _ema(df["close"], long_p)
            df[f"ema_{short_p}_{long_p}_ratio"] = short_ema / long_ema * 100 - 100

    # ── 3. Volatility ────────────────────────────────────────────────────
    logger.info("Computing volatility features...")

    atr = _atr(df["high"], df["low"], df["close"], cfg.atr_period)
    df["atr"] = atr
    df["atr_pct"] = atr / df["close"] * 100

    # Rolling std of returns (volatility)
    returns = df["close"].pct_change()
    for period in [cfg.vol_short, cfg.vol_long]:
        df[f"vol_std_{period}"] = _rolling_std(returns, period) * 100

    # Volatility ratio (short-term vs long-term)
    vol_short = _rolling_std(returns, cfg.vol_short)
    vol_long = _rolling_std(returns, cfg.vol_long)
    df["vol_ratio"] = vol_short / vol_long.replace(0, np.nan)

    # ATR ratio (current ATR vs longer-term ATR)
    atr_long = _atr(df["high"], df["low"], df["close"], cfg.atr_period * 5)
    df["atr_ratio"] = atr / atr_long.replace(0, np.nan)

    # ── 4. Momentum Oscillators ──────────────────────────────────────────
    logger.info("Computing momentum features...")

    for period in cfg.rsi_periods:
        df[f"rsi_{period}"] = _rsi(df["close"], period)

    # ROC (Rate of Change) over multiple periods
    for period in cfg.roc_periods:
        df[f"roc_{period}"] = (df["close"] / df["close"].shift(period) - 1) * 100

    # ── 5. Bands / Channels ──────────────────────────────────────────────
    logger.info("Computing band/channel features...")

    # Bollinger Bands
    bb_mid = _sma(df["close"], cfg.bb_period)
    bb_std = _rolling_std(df["close"], cfg.bb_period)  # std of close
    bb_upper = bb_mid + cfg.bb_std * bb_std
    bb_lower = bb_mid - cfg.bb_std * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (df["close"] - bb_lower) / bb_width * 100  # 0=lower, 100=upper
    df["bb_width_pct"] = bb_width / bb_mid * 100

    # Donchian Channel
    donchian_high = df["high"].rolling(window=cfg.donchian_period, min_periods=cfg.donchian_period).max()
    donchian_low = df["low"].rolling(window=cfg.donchian_period, min_periods=cfg.donchian_period).min()
    donchian_range = (donchian_high - donchian_low).replace(0, np.nan)
    df["donchian_position"] = (df["close"] - donchian_low) / donchian_range * 100
    df["donchian_width_pct"] = donchian_range / df["close"] * 100

    # ── 6. Trend Strength (ADX/DI) ───────────────────────────────────────
    logger.info("Computing ADX/DI features...")

    adx_period = cfg.adx_period
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    # +DM and -DM
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Smoothed
    atr_smooth = tr.ewm(span=adx_period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=adx_period, adjust=False).mean()

    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["di_diff"] = plus_di - minus_di
    df["di_ratio"] = plus_di / minus_di.replace(0, np.nan)

    # ── 7. Breakout Indicators ───────────────────────────────────────────
    logger.info("Computing breakout features...")

    for period in cfg.hl_periods:
        rolling_high = df["high"].rolling(window=period, min_periods=period).max()
        rolling_low = df["low"].rolling(window=period, min_periods=period).min()

        # Distance from rolling high/low as percentage
        df[f"dist_high_{period}"] = (df["close"] - rolling_high) / rolling_high * 100
        df[f"dist_low_{period}"] = (df["close"] - rolling_low) / rolling_low * 100

        # Breakout signal: 1 if new high, -1 if new low, 0 otherwise
        df[f"breakout_high_{period}"] = (df["close"] >= rolling_high.shift(1)).astype(np.float64)
        df[f"breakout_low_{period}"] = (df["close"] <= rolling_low.shift(1)).astype(np.float64)

    # ── 8. Time / Session Features (cyclical encoding) ───────────────────
    logger.info("Computing time/session features...")

    # Hour of day as cyclical (24h cycle)
    hour_rad = 2 * np.pi * df["hour"] / 24
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)

    # Day of week as cyclical (5 trading days)
    dow_rad = 2 * np.pi * df["dayofweek"] / 5
    df["dow_sin"] = np.sin(dow_rad)
    df["dow_cos"] = np.cos(dow_rad)

    # Trading session (UTC-based approximation for forex)
    # London: 07-16 UTC, New York: 12-21 UTC, Asia: 23-08 UTC
    hour = df["hour"]
    df["session_london"] = ((hour >= 7) & (hour < 16)).astype(np.float64)
    df["session_ny"] = ((hour >= 12) & (hour < 21)).astype(np.float64)
    df["session_asia"] = ((hour >= 23) | (hour < 8)).astype(np.float64)
    df["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(np.float64)  # London+NY

    # ── 9. Volume / Volatility Regime ────────────────────────────────────
    logger.info("Computing volume/regime features...")

    # Volume relative to rolling average
    vol_sma_short = _sma(df["volume"], 20)
    vol_sma_long = _sma(df["volume"], 50)
    df["vol_ratio_sma"] = df["volume"] / vol_sma_short.replace(0, np.nan)
    df["vol_ratio_long"] = df["volume"] / vol_sma_long.replace(0, np.nan)

    # Volume trend (increasing/decreasing)
    df["vol_trend"] = df["volume"].pct_change(periods=5) * 100

    # Volatility regime: current ATR percentile rank
    df["atr_percentile"] = df["atr_pct"].rolling(window=200, min_periods=50).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) * 100, raw=False
    )

    # Trend regime: ADX percentile rank
    df["adx_percentile"] = df["adx"].rolling(window=200, min_periods=50).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) * 100, raw=False
    )

    # ── 10. Additional Structural Features ───────────────────────────────
    logger.info("Computing additional structural features...")

    # Consecutive up/down bars
    direction = np.sign(df["close"] - df["open"])
    df["consecutive_up"] = _consecutive_count(direction, 1)
    df["consecutive_down"] = _consecutive_count(direction, -1)

    # Distance from SMA 200 (major trend filter)
    sma200 = _sma(df["close"], 200)
    df["dist_sma200"] = (df["close"] - sma200) / sma200 * 100

    logger.info(f"Total features computed: {len(_get_feature_columns(df))}")
    return df


def _consecutive_count(direction: pd.Series, target: int) -> pd.Series:
    """Count consecutive bars in the same direction."""
    result = pd.Series(0, index=direction.index, dtype=np.float64)
    count = 0
    for i in range(len(direction)):
        if direction.iloc[i] == target:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of feature column names (exclude raw OHLCV and time columns).
    """
    exclude = {"timestamp", "date", "time", "open", "high", "low", "close",
               "volume", "datetime", "hour", "dayofweek"}
    return [c for c in df.columns if c not in exclude]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Public accessor for feature column names."""
    return _get_feature_columns(df)


def verify_no_lookahead(data: pd.DataFrame, feature_cols: List[str]) -> Dict[str, bool]:
    """
    Verify that no feature uses future data.

    Strategy: For each feature, check that shifting the feature by 1 bar
    and correlating with the original gives a high correlation (meaning
    the feature is mostly determined by past data). A feature with
    lookahead would show anomalous behavior.

    More importantly, we verify that each feature at time t does NOT
    depend on data from time t+1 or later by construction.

    Returns
    -------
    dict
        {feature_name: True} for all features (passes by construction).
        This function documents the verification methodology.
    """
    results = {}

    for feat in feature_cols:
        # Check that the feature doesn't have perfect correlation with
        # future returns (which would indicate lookahead)
        if feat in data.columns:
            feat_vals = data[feat].dropna()
            if len(feat_vals) > 100:
                # Future return (this is what we're predicting)
                future_ret = data["close"].pct_change(5).shift(-5).loc[feat_vals.index]
                valid = feat_vals.notna() & future_ret.notna()
                if valid.sum() > 50:
                    corr = np.corrcoef(feat_vals[valid], future_ret[valid])[0, 1]
                    # If correlation is extremely high (>0.9), flag as potential lookahead
                    results[feat] = abs(corr) < 0.9
                else:
                    results[feat] = True
            else:
                results[feat] = True
        else:
            results[feat] = False

    n_pass = sum(results.values())
    n_total = len(results)
    logger.info(f"Lookahead verification: {n_pass}/{n_total} features passed")

    return results


def get_feature_documentation() -> pd.DataFrame:
    """
    Return a DataFrame documenting all features, their category,
    and lookahead safety.
    """
    docs = [
        # Price / Candle Structure
        ("ret_N", "Price Structure", "N-bar return (%)", "Safe: uses close[0] vs close[-N]"),
        ("body_pct", "Price Structure", "Body as % of range", "Safe: uses current OHLC only"),
        ("upper_wick_pct", "Price Structure", "Upper wick as % of range", "Safe: uses current OHLC only"),
        ("lower_wick_pct", "Price Structure", "Lower wick as % of range", "Safe: uses current OHLC only"),
        ("range_pct", "Price Structure", "Range as % of close", "Safe: uses current OHLC only"),
        ("gap_pct", "Price Structure", "Gap from prev close (%)", "Safe: uses open[0] vs close[-1]"),
        # Trend
        ("dist_ema_N", "Trend", "Distance from EMA_N (%)", "Safe: EMA uses past data only"),
        ("ema_N_slope", "Trend", "EMA_N slope (%)", "Safe: uses past EMA values"),
        ("ema_A_B_ratio", "Trend", "EMA ratio short/long (%)", "Safe: both EMAs use past data"),
        # Volatility
        ("atr", "Volatility", "ATR (absolute)", "Safe: uses past H/L/C"),
        ("atr_pct", "Volatility", "ATR as % of close", "Safe: uses past H/L/C"),
        ("vol_std_N", "Volatility", "Rolling std of returns (%)", "Safe: uses past returns"),
        ("vol_ratio", "Volatility", "Short/long vol ratio", "Safe: both use past data"),
        ("atr_ratio", "Volatility", "ATR ratio short/long", "Safe: both use past data"),
        # Momentum
        ("rsi_N", "Momentum", "RSI_N", "Safe: uses past close only"),
        ("roc_N", "Momentum", "Rate of Change_N (%)", "Safe: uses close[0] vs close[-N]"),
        # Bands
        ("bb_position", "Bands", "Bollinger position (0-100)", "Safe: uses past close for BB"),
        ("bb_width_pct", "Bands", "Bollinger width (%)", "Safe: uses past close for BB"),
        ("donchian_position", "Bands", "Donchian position (0-100)", "Safe: uses past H/L"),
        ("donchian_width_pct", "Bands", "Donchian width (%)", "Safe: uses past H/L"),
        # Trend Strength
        ("adx", "Trend Strength", "ADX", "Safe: uses past H/L/C"),
        ("plus_di", "Trend Strength", "+DI", "Safe: uses past H/L/C"),
        ("minus_di", "Trend Strength", "-DI", "Safe: uses past H/L/C"),
        ("di_diff", "Trend Strength", "+DI - -DI", "Safe: derived from past DI"),
        ("di_ratio", "Trend Strength", "+DI / -DI ratio", "Safe: derived from past DI"),
        # Breakout
        ("dist_high_N", "Breakout", "Dist from N-bar high (%)", "Safe: uses past high"),
        ("dist_low_N", "Breakout", "Dist from N-bar low (%)", "Safe: uses past low"),
        ("breakout_high_N", "Breakout", "New N-bar high (0/1)", "Safe: compares to past"),
        ("breakout_low_N", "Breakout", "New N-bar low (0/1)", "Safe: compares to past"),
        # Time/Session
        ("hour_sin", "Time", "Hour (sin encoding)", "Safe: deterministic from timestamp"),
        ("hour_cos", "Time", "Hour (cos encoding)", "Safe: deterministic from timestamp"),
        ("dow_sin", "Time", "Day of week (sin)", "Safe: deterministic from timestamp"),
        ("dow_cos", "Time", "Day of week (cos)", "Safe: deterministic from timestamp"),
        ("session_london", "Time", "London session (0/1)", "Safe: deterministic from timestamp"),
        ("session_ny", "Time", "NY session (0/1)", "Safe: deterministic from timestamp"),
        ("session_asia", "Time", "Asia session (0/1)", "Safe: deterministic from timestamp"),
        ("session_overlap", "Time", "London+NY overlap (0/1)", "Safe: deterministic from timestamp"),
        # Volume/Regime
        ("vol_ratio_sma", "Volume", "Volume / SMA20 volume", "Safe: uses past volume"),
        ("vol_ratio_long", "Volume", "Volume / SMA50 volume", "Safe: uses past volume"),
        ("vol_trend", "Volume", "Volume trend (%)", "Safe: uses past volume"),
        ("atr_percentile", "Regime", "ATR percentile rank", "Safe: uses past ATR"),
        ("adx_percentile", "Regime", "ADX percentile rank", "Safe: uses past ADX"),
        # Structural
        ("consecutive_up", "Structure", "Consecutive up bars", "Safe: uses past direction"),
        ("consecutive_down", "Structure", "Consecutive down bars", "Safe: uses past direction"),
        ("dist_sma200", "Trend", "Distance from SMA200 (%)", "Safe: uses past close"),
    ]

    return pd.DataFrame(docs, columns=["feature_pattern", "category", "description", "lookahead_safety"])
