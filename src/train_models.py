#!/usr/bin/env python3
"""
XGBoost Training Pipeline for 4 Trading Models:
1. Volatilitäts-Regime (LOW/NORMAL/HIGH) - Multiclass
2. Future Range - Regression → AUC via direction classification
3. Large-Candle Probability - Binary Classification
4. Trend-/Range-Regime - Binary Classification
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

data_files = [
    "XAUUSD_M5_2024-08.csv", "XAUUSD_M5_2024-09.csv", "XAUUSD_M5_2024-10.csv",
    "XAUUSD_M5_2024-11.csv", "XAUUSD_M5_2024-12.csv", "XAUUSD_M5_2025-01.csv",
    "XAUUSD_M5_2025-02.csv", "XAUUSD_M5_2025-03.csv", "XAUUSD_M5_2025-04.csv",
    "XAUUSD_M5_2025-05.csv", "XAUUSD_M5_2025-06.csv", "XAUUSD_M5_2025-07.csv",
    "XAUUSD_M5_2025-08.csv", "XAUUSD_M5_2025-09.csv", "XAUUSD_M5_2025-10.csv",
    "XAUUSD_M5_2025-11.csv", "XAUUSD_M5_2025-12.csv", "XAUUSD_M5_2026-01.csv",
    "XAUUSD_M5_2026-02.csv", "XAUUSD_M5_2026-03.csv", "XAUUSD_M5_2026-04.csv",
    "XAUUSD_M5_2026-05.csv", "XAUUSD_M5_2026-06.csv", "XAUUSD_M5_2026-07.csv",
    "XAUUSD_M5_2026-08.csv",
]

dfs = []
for f in data_files:
    tmp = pd.read_csv(f)
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values('datetime').reset_index(drop=True)
print(f"Loaded {len(df)} bars | Date range: {df['datetime'].min()} → {df['datetime'].max()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

def engineer_features(df):
    """Create comprehensive feature set for all 4 models."""
    d = df.copy()

    # --- Price Structure ---
    d['returns'] = d['close'].pct_change()
    d['log_returns'] = np.log(d['close'] / d['close'].shift(1))
    d['range'] = d['high'] - d['low']
    d['body'] = abs(d['close'] - d['open'])
    d['upper_wick'] = d['high'] - d[['open', 'close']].max(axis=1)
    d['lower_wick'] = d[['open', 'close']].min(axis=1) - d['low']
    d['body_ratio'] = d['body'] / (d['range'] + 1e-10)
    d['close_position'] = (d['close'] - d['low']) / (d['range'] + 1e-10)

    # --- ATR (multiple periods) ---
    for period in [5, 10, 20, 50]:
        d[f'atr_{period}'] = d['range'].rolling(period).mean()

    # ATR ratios
    d['atr_ratio_5_20'] = d['atr_5'] / (d['atr_20'] + 1e-10)
    d['atr_ratio_10_50'] = d['atr_10'] / (d['atr_50'] + 1e-10)
    d['atr_normalized'] = d['atr_20'] / d['close'] * 10000  # in pips equivalent

    # --- Garman-Klass Volatility ---
    d['gk_vol'] = np.sqrt(
        0.5 * np.log(d['high'] / d['low']) ** 2 -
        (2 * np.log(2) - 1) * np.log(d['close'] / d['open']) ** 2
    )
    d['gk_vol_10'] = d['gk_vol'].rolling(10).mean()
    d['gk_vol_50'] = d['gk_vol'].rolling(50).mean()

    # --- Volatility of Volatility ---
    d['vol_of_vol'] = d['atr_20'].rolling(20).std()
    d['atr_momentum'] = d['atr_20'] - d['atr_20'].shift(5)

    # --- Moving Averages & Distance ---
    for period in [5, 10, 20, 50, 100, 200]:
        d[f'sma_{period}'] = d['close'].rolling(period).mean()
    for period in [5, 10, 20, 50, 100, 200]:
        atr_ref = d[f'atr_{period}'] if f'atr_{period}' in d.columns else d['atr_50']
        d[f'dist_sma_{period}'] = (d['close'] - d[f'sma_{period}']) / (atr_ref + 1e-10)

    # MA crossovers
    d['sma_5_20_cross'] = (d['sma_5'] - d['sma_20']) / (d['atr_20'] + 1e-10)
    d['sma_20_50_cross'] = (d['sma_20'] - d['sma_50']) / (d['atr_20'] + 1e-10)
    d['sma_50_200_cross'] = (d['sma_50'] - d['sma_200']) / (d['atr_50'] + 1e-10)

    # --- Trend Strength (ADX-like) ---
    for period in [10, 20]:
        plus_dm = d['high'].diff()
        minus_dm = -d['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        atr_sum = d['range'].rolling(period).sum()
        plus_di = plus_dm.rolling(period).sum() / (atr_sum + 1e-10) * 100
        minus_di = minus_dm.rolling(period).sum() / (atr_sum + 1e-10) * 100
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        d[f'adx_{period}'] = dx.rolling(period).mean()
        d[f'plus_di_{period}'] = plus_di
        d[f'minus_di_{period}'] = minus_di

    # --- Momentum / RSI ---
    for period in [5, 10, 14, 20]:
        delta = d['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        d[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = d['close'].ewm(span=12).mean()
    ema26 = d['close'].ewm(span=26).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']

    # --- Bollinger Bands ---
    for period in [20, 50]:
        sma = d['close'].rolling(period).mean()
        std = d['close'].rolling(period).std()
        d[f'bb_upper_{period}'] = sma + 2 * std
        d[f'bb_lower_{period}'] = sma - 2 * std
        d[f'bb_width_{period}'] = (4 * std) / (sma + 1e-10)
        d[f'bb_position_{period}'] = (d['close'] - d[f'bb_lower_{period}']) / (4 * std + 1e-10)

    # --- Stochastic ---
    for period in [10, 14]:
        low_min = d['low'].rolling(period).min()
        high_max = d['high'].rolling(period).max()
        d[f'stoch_k_{period}'] = 100 * (d['close'] - low_min) / (high_max - low_min + 1e-10)
        d[f'stoch_d_{period}'] = d[f'stoch_k_{period}'].rolling(3).mean()

    # --- Returns over multiple horizons ---
    for h in [1, 2, 3, 5, 10, 15, 20, 30]:
        d[f'ret_{h}bar'] = d['close'].pct_change(h)
        d[f'range_{h}bar'] = d['close'].rolling(h).apply(
            lambda x: (x.max() - x.min()) / (x.mean() + 1e-10), raw=True
        )

    # --- Skewness & Kurtosis ---
    for period in [10, 20, 50]:
        d[f'skew_{period}'] = d['returns'].rolling(period).skew()
        d[f'kurt_{period}'] = d['returns'].rolling(period).kurt()

    # --- Volume Features ---
    d['volume_sma_20'] = d['volume'].rolling(20).mean()
    d['volume_ratio'] = d['volume'] / (d['volume_sma_20'] + 1e-10)
    d['volume_change'] = d['volume'].pct_change(5)

    # --- Time Features ---
    d['hour'] = d['datetime'].dt.hour
    d['minute'] = d['datetime'].dt.minute
    d['day_of_week'] = d['datetime'].dt.dayofweek
    d['is_london'] = ((d['hour'] >= 7) & (d['hour'] <= 16)).astype(int)
    d['is_ny'] = ((d['hour'] >= 12) & (d['hour'] <= 21)).astype(int)
    d['is_overlap'] = ((d['hour'] >= 12) & (d['hour'] <= 16)).astype(int)

    # --- Consecutive moves ---
    d['consec_up'] = (d['returns'] > 0).astype(int).groupby(
        (d['returns'] <= 0).astype(int).cumsum()
    ).cumsum()
    d['consec_down'] = (d['returns'] < 0).astype(int).groupby(
        (d['returns'] >= 0).astype(int).cumsum()
    ).cumsum()

    # --- Drawdown from recent high ---
    rolling_high = d['high'].rolling(50).max()
    d['drawdown'] = (d['close'] - rolling_high) / (rolling_high + 1e-10)

    return d

df = engineer_features(df)
print(f"Features engineered. Shape: {df.shape}")

# ============================================================
# 3. LABEL CREATION
# ============================================================
print("\n" + "=" * 70)
print("LABEL CREATION")
print("=" * 70)

# --- Model 1: Volatilitäts-Regime (LOW/NORMAL/HIGH) ---
# Forward-looking volatility over next 10 bars
future_atr = df['range'].rolling(10).mean().shift(-10)
vol_p33 = future_atr.quantile(0.33)
vol_p66 = future_atr.quantile(0.66)

def classify_vol(x):
    if pd.isna(x):
        return np.nan
    if x <= vol_p33:
        return 0  # LOW
    elif x <= vol_p66:
        return 1  # NORMAL
    else:
        return 2  # HIGH

df['label_vol_regime'] = future_atr.apply(classify_vol)
print(f"Vol-Regime distribution: {df['label_vol_regime'].value_counts().to_dict()}")
print(f"  Thresholds: LOW≤{vol_p33:.4f} | NORMAL≤{vol_p66:.4f} | HIGH>{vol_p66:.4f}")

# --- Model 2: Future Range (Regression → Binary for AUC) ---
# How large will the range be in next 10 bars?
future_range_10 = df['close'].rolling(10).apply(
    lambda x: (x.max() - x.min()), raw=True
).shift(-10)
future_range_atr_norm = future_range_10 / (df['atr_20'] + 1e-10)
df['label_future_range'] = future_range_atr_norm
# Binary: above median range = 1 (large move), below = 0
range_median = df['label_future_range'].median()
df['label_future_range_binary'] = (df['label_future_range'] > range_median).astype(int)
print(f"\nFuture Range: median={range_median:.3f} ATR")
print(f"  Binary distribution: {df['label_future_range_binary'].value_counts().to_dict()}")

# --- Model 3: Large-Candle Probability ---
# Will there be a candle > 2x ATR in next 5 bars?
future_max_body = df['body'].rolling(5).max().shift(-5)
df['label_large_candle'] = (future_max_body > 2 * df['atr_20']).astype(int)
print(f"\nLarge Candle distribution: {df['label_large_candle'].value_counts().to_dict()}")
print(f"  Positive rate: {df['label_large_candle'].mean():.3f}")

# --- Model 4: Trend-/Range-Regime ---
# ADX-based: ADX > 25 = Trend, ADX < 20 = Range
adx_forward = df['adx_20'].shift(-10)  # future ADX as proxy
# Also use efficiency ratio
def efficiency_ratio(prices):
    """Close-to-close efficiency ratio."""
    change = abs(prices.iloc[-1] - prices.iloc[0])
    path = prices.diff().abs().sum()
    return change / (path + 1e-10)

df['efficiency_ratio'] = df['close'].rolling(20).apply(efficiency_ratio, raw=False)
df['efficiency_ratio'] = df['efficiency_ratio'].shift(-10)  # forward-looking

# Label: High efficiency ratio + high ADX = Trend (1), else Range (0)
er_median = df['efficiency_ratio'].median()
adx_threshold = 25
df['label_trend_regime'] = ((df['efficiency_ratio'] > er_median) & (adx_forward > adx_threshold)).astype(int)
print(f"\nTrend/Range Regime distribution: {df['label_trend_regime'].value_counts().to_dict()}")
print(f"  Trend rate: {df['label_trend_regime'].mean():.3f}")

# ============================================================
# 4. PREPARE FEATURE MATRIX
# ============================================================
print("\n" + "=" * 70)
print("PREPARING FEATURE MATRIX")
print("=" * 70)

# Exclude non-feature columns
exclude_cols = [
    'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume',
    'datetime', 'label_vol_regime', 'label_future_range', 'label_future_range_binary',
    'label_large_candle', 'label_trend_regime', 'returns', 'log_returns'
]

feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

# Drop rows with NaN in features or labels
label_cols = ['label_vol_regime', 'label_future_range_binary', 'label_large_candle', 'label_trend_regime']
valid_mask = df[feature_cols + label_cols].notna().all(axis=1)
df_clean = df[valid_mask].reset_index(drop=True)
print(f"Valid rows after NaN removal: {len(df_clean)}")

X = df_clean[feature_cols].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# 5. WALK-FORWARD TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD TRAINING & EVALUATION")
print("=" * 70)

# Time-series split: 5 folds
n = len(df_clean)
fold_size = n // 5
folds = []
for i in range(5):
    test_start = i * fold_size
    test_end = min((i + 1) * fold_size, n)
    train_end = max(0, test_start - 100)  # 100 bar embargo
    folds.append((0, train_end, test_start, test_end))

# XGBoost params
xgb_params_binary = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_weight': 50,
    'n_estimators': 500,
    'verbosity': 0,
    'random_state': 42,
}

xgb_params_multi = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_weight': 50,
    'n_estimators': 500,
    'verbosity': 0,
    'random_state': 42,
}

def train_and_evaluate(X, y, model_name, params, is_multiclass=False):
    """Walk-forward training and evaluation."""
    aucs = []
    f1s = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        if len(np.unique(y_train)) < 2:
            continue

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        if is_multiclass:
            y_pred_proba = model.predict_proba(X_test)
            # One-vs-Rest AUC for multiclass
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc = 0.5
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        aucs.append(auc)
        f1s.append(f1)
        print(f"  Fold {fold_idx+1}: AUC={auc:.4f} | F1={f1:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_f1 = np.mean(f1s)
    return mean_auc, std_auc, mean_f1

# ============================================================
# 6. TRAIN ALL 4 MODELS
# ============================================================
results = {}

# --- Model 1: Volatilitäts-Regime ---
print("\n" + "-" * 50)
print("MODEL 1: VOLATILITÄTS-REGIME (LOW/NORMAL/HIGH)")
print("-" * 50)
y1 = df_clean['label_vol_regime'].values.astype(int)
auc1, std1, f1_1 = train_and_evaluate(X, y1, "Vol-Regime", xgb_params_multi, is_multiclass=True)
results['Volatilitäts-Regime'] = (auc1, std1, f1_1)
print(f"  → Mean AUC: {auc1:.4f} ± {std1:.4f} | F1: {f1_1:.4f}")

# --- Model 2: Future Range ---
print("\n" + "-" * 50)
print("MODEL 2: FUTURE RANGE (Large Move Probability)")
print("-" * 50)
y2 = df_clean['label_future_range_binary'].values.astype(int)
auc2, std2, f1_2 = train_and_evaluate(X, y2, "Future-Range", xgb_params_binary)
results['Future Range'] = (auc2, std2, f1_2)
print(f"  → Mean AUC: {auc2:.4f} ± {std2:.4f} | F1: {f1_2:.4f}")

# --- Model 3: Large-Candle Probability ---
print("\n" + "-" * 50)
print("MODEL 3: LARGE-CANDLE PROBABILITY")
print("-" * 50)
y3 = df_clean['label_large_candle'].values.astype(int)
auc3, std3, f1_3 = train_and_evaluate(X, y3, "Large-Candle", xgb_params_binary)
results['Large-Candle'] = (auc3, std3, f1_3)
print(f"  → Mean AUC: {auc3:.4f} ± {std3:.4f} | F1: {f1_3:.4f}")

# --- Model 4: Trend-/Range-Regime ---
print("\n" + "-" * 50)
print("MODEL 4: TREND-/RANGE-REGIME")
print("-" * 50)
y4 = df_clean['label_trend_regime'].values.astype(int)
auc4, std4, f1_4 = train_and_evaluate(X, y4, "Trend-Range", xgb_params_binary)
results['Trend/Range-Regime'] = (auc4, std4, f1_4)
print(f"  → Mean AUC: {auc4:.4f} ± {std4:.4f} | F1: {f1_4:.4f}")

# ============================================================
# 7. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<30} {'AUC':>10} {'± Std':>8} {'F1':>8}")
print("-" * 60)
for name, (auc, std, f1) in results.items():
    stars = "⭐" * 5
    print(f"{name:<30} {auc:>10.4f} {std:>8.4f} {f1:>8.4f}  {stars}")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("  AUC = 0.50 → Random (no predictive power)")
print("  AUC = 0.55 → Weak but usable signal")
print("  AUC = 0.60 → Moderate predictive power")
print("  AUC = 0.65+ → Strong signal")
print("  AUC = 0.70+ → Very strong (rare in finance)")
print("=" * 70)

# ============================================================
# 8. FEATURE IMPORTANCE (Top 10 per model)
# ============================================================
print("\n" + "=" * 70)
print("TOP 10 FEATURES PER MODEL")
print("=" * 70)

def get_top_features(X, y, params, model_name, is_multiclass=False, top_n=10):
    """Train on full data and get feature importance."""
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:top_n]
    print(f"\n{model_name}:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[idx]:<30} {importance[idx]:.4f}")

get_top_features(X, y1, xgb_params_multi, "Volatilitäts-Regime", is_multiclass=True)
get_top_features(X, y2, xgb_params_binary, "Future Range")
get_top_features(X, y3, xgb_params_binary, "Large-Candle Probability")
get_top_features(X, y4, xgb_params_binary, "Trend/Range-Regime")

print("\n✅ Training complete!")


# ============================================================
# 9. DEDICATED OUT-OF-SAMPLE (OOS) HOLDOUT TEST
# ============================================================
print("\n" + "=" * 70)
print("DEDICATED OOS HOLDOUT TEST")
print("=" * 70)

# Split: Train on data before 2026-07-01, Test on Jul-Aug 2026
oos_date = pd.Timestamp('2026-07-01')
train_mask = df_clean['datetime'] < oos_date
test_mask = df_clean['datetime'] >= oos_date

X_train_oos = X[train_mask.values]
X_test_oos = X[test_mask.values]
print(f"Training period: {df_clean.loc[train_mask, 'datetime'].min()} → {df_clean.loc[train_mask, 'datetime'].max()} ({train_mask.sum()} bars)")
print(f"OOS Test period: {df_clean.loc[test_mask, 'datetime'].min()} → {df_clean.loc[test_mask, 'datetime'].max()} ({test_mask.sum()} bars)")

oos_results = {}

# --- OOS Model 1 ---
print("\n" + "-" * 50)
print("OOS MODEL 1: VOLATILITÄTS-REGIME")
print("-" * 50)
y1_train = y1[train_mask.values]
y1_test = y1[test_mask.values]
model_oos1 = xgb.XGBClassifier(**xgb_params_multi)
model_oos1.fit(X_train_oos, y1_train, verbose=False)
y1_pred_proba = model_oos1.predict_proba(X_test_oos)
try:
    auc_oos1 = roc_auc_score(y1_test, y1_pred_proba, multi_class='ovr', average='weighted')
except:
    auc_oos1 = 0.5
y1_pred = model_oos1.predict(X_test_oos)
f1_oos1 = f1_score(y1_test, y1_pred, average='weighted', zero_division=0)
oos_results['Volatilitäts-Regime'] = (auc_oos1, f1_oos1)
print(f"  OOS AUC: {auc_oos1:.4f} | F1: {f1_oos1:.4f}")

# --- OOS Model 2 ---
print("\n" + "-" * 50)
print("OOS MODEL 2: FUTURE RANGE")
print("-" * 50)
y2_train = y2[train_mask.values]
y2_test = y2[test_mask.values]
model_oos2 = xgb.XGBClassifier(**xgb_params_binary)
model_oos2.fit(X_train_oos, y2_train, verbose=False)
y2_pred_proba = model_oos2.predict_proba(X_test_oos)[:, 1]
try:
    auc_oos2 = roc_auc_score(y2_test, y2_pred_proba)
except:
    auc_oos2 = 0.5
y2_pred = model_oos2.predict(X_test_oos)
f1_oos2 = f1_score(y2_test, y2_pred, average='weighted', zero_division=0)
oos_results['Future Range'] = (auc_oos2, f1_oos2)
print(f"  OOS AUC: {auc_oos2:.4f} | F1: {f1_oos2:.4f}")

# --- OOS Model 3 ---
print("\n" + "-" * 50)
print("OOS MODEL 3: LARGE-CANDLE PROBABILITY")
print("-" * 50)
y3_train = y3[train_mask.values]
y3_test = y3[test_mask.values]
model_oos3 = xgb.XGBClassifier(**xgb_params_binary)
model_oos3.fit(X_train_oos, y3_train, verbose=False)
y3_pred_proba = model_oos3.predict_proba(X_test_oos)[:, 1]
try:
    auc_oos3 = roc_auc_score(y3_test, y3_pred_proba)
except:
    auc_oos3 = 0.5
y3_pred = model_oos3.predict(X_test_oos)
f1_oos3 = f1_score(y3_test, y3_pred, average='weighted', zero_division=0)
oos_results['Large-Candle'] = (auc_oos3, f1_oos3)
print(f"  OOS AUC: {auc_oos3:.4f} | F1: {f1_oos3:.4f}")

# --- OOS Model 4 ---
print("\n" + "-" * 50)
print("OOS MODEL 4: TREND-/RANGE-REGIME")
print("-" * 50)
y4_train = y4[train_mask.values]
y4_test = y4[test_mask.values]
model_oos4 = xgb.XGBClassifier(**xgb_params_binary)
model_oos4.fit(X_train_oos, y4_train, verbose=False)
y4_pred_proba = model_oos4.predict_proba(X_test_oos)[:, 1]
try:
    auc_oos4 = roc_auc_score(y4_test, y4_pred_proba)
except:
    auc_oos4 = 0.5
y4_pred = model_oos4.predict(X_test_oos)
f1_oos4 = f1_score(y4_test, y4_pred, average='weighted', zero_division=0)
oos_results['Trend/Range-Regime'] = (auc_oos4, f1_oos4)
print(f"  OOS AUC: {auc_oos4:.4f} | F1: {f1_oos4:.4f}")

# ============================================================
# 10. OOS vs WALK-FORWARD COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("OOS vs WALK-FORWARD COMPARISON")
print("=" * 70)
print(f"{'Model':<30} {'WF AUC':>10} {'OOS AUC':>10} {'Δ':>8} {'OOS F1':>8}")
print("-" * 70)
for name in results:
    wf_auc = results[name][0]
    oos_auc = oos_results[name][0]
    oos_f1 = oos_results[name][1]
    delta = oos_auc - wf_auc
    sign = "+" if delta >= 0 else ""
    print(f"{name:<30} {wf_auc:>10.4f} {oos_auc:>10.4f} {sign}{delta:>7.4f} {oos_f1:>8.4f}")

print("\n" + "=" * 70)
print("LEGEND:")
print("  WF = Walk-Forward (5-fold cross-validation)")
print("  OOS = Out-of-Sample (trained on data before Jul 2026, tested on Jul-Aug 2026)")
print("  Δ = Difference (OOS - WF). Negative = possible overfitting.")
print("  If OOS AUC drops >0.05 from WF → overfitting risk")
print("=" * 70)

# --- OOS Classification Reports ---
print("\n" + "=" * 70)
print("OOS DETAILED CLASSIFICATION REPORTS")
print("=" * 70)

print("\n--- Volatilitäts-Regime ---")
print(classification_report(y1_test, y1_pred, target_names=['LOW', 'NORMAL', 'HIGH'], zero_division=0))

print("\n--- Future Range ---")
print(classification_report(y2_test, y2_pred, target_names=['Small Range', 'Large Range'], zero_division=0))

print("\n--- Large-Candle Probability ---")
print(classification_report(y3_test, y3_pred, target_names=['No Big Candle', 'Big Candle'], zero_division=0))

print("\n--- Trend/Range-Regime ---")
print(classification_report(y4_test, y4_pred, target_names=['Range', 'Trend'], zero_division=0))

print("\n✅ Complete analysis finished!")
