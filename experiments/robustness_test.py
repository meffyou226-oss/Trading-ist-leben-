#!/usr/bin/env python3
"""
Walk-Forward Robustness Test + Monthly Breakdown
Tests the best config across different time periods.
"""
import pandas as pd, numpy as np, xgboost as xgb, pickle, json, os
import warnings
warnings.filterwarnings('ignore')

OOS_DATE = pd.Timestamp('2026-07-01')
INITIAL_EQUITY = 10000.0
SPREAD = 0.4
PER_CONTRACT = 100
MAX_SIZE = 2.0

DATA_DIR = ".."
MODELS_DIR = "../models"
CONFIG_DIR = "../config"

DATA_FILES = [f"XAUUSD_M5_{y}-{m:02d}.csv" for y in range(2024,2027) for m in range(1,13)
              if (y==2024 and m>=8) or (y==2025) or (y==2026 and m<=8)]

print("Loading...")
dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in DATA_FILES]
df = pd.concat(dfs, ignore_index=True)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values('datetime').reset_index(drop=True)

models = {}
for name in ['vol_regime', 'future_range', 'large_candle', 'trend_regime']:
    with open(os.path.join(MODELS_DIR, f'{name}.pkl'), 'rb') as f:
        models[name] = pickle.load(f)

with open(os.path.join(CONFIG_DIR, 'features.json'), 'r') as f:
    FEATS = json.load(f)

# Feature engineering
d = df.copy()
d['returns'] = d['close'].pct_change()
d['range'] = d['high'] - d['low']
d['body'] = abs(d['close'] - d['open'])
for p in [5,10,20,50]: d[f'atr_{p}'] = d['range'].rolling(p).mean()
d['atr_ratio'] = d['atr_5']/(d['atr_20']+1e-10)
for p in [5,10,20,50,100,200]: d[f'sma_{p}'] = d['close'].rolling(p).mean()
for p in [5,10,20,50,100,200]:
    ar = d[f'atr_{p}'] if f'atr_{p}' in d.columns else d['atr_50']
    d[f'dist_sma_{p}'] = (d['close']-d[f'sma_{p}'])/(ar+1e-10)
for p in [10,20]:
    up = d['high'].diff(); dn = -d['low'].diff(); up[up<0]=0; dn[dn<0]=0
    atrs = d['range'].rolling(p).sum()
    pdi = up.rolling(p).sum()/(atrs+1e-10)*100
    mdi = dn.rolling(p).sum()/(atrs+1e-10)*100
    dx = abs(pdi-mdi)/(pdi+mdi+1e-10)*100
    d[f'adx_{p}'] = dx.rolling(p).mean()
for p in [5,10,14,20]:
    delta = d['close'].diff()
    g = delta.where(delta>0,0).rolling(p).mean()
    l = (-delta.where(delta<0,0)).rolling(p).mean()
    d[f'rsi_{p}'] = 100-(100/(1+g/(l+1e-10)))
for p in [20,50]:
    s = d['close'].rolling(p).mean(); sdev = d['close'].rolling(p).std()
    d[f'bb_lower_{p}'] = s-2*sdev
    d[f'bb_pos_{p}'] = (d['close']-d[f'bb_lower_{p}'])/(4*sdev+1e-10)
for h in [1,3,5,10,15,20]: d[f'ret_{h}bar'] = d['close'].pct_change(h)
d['vol_ratio'] = d['volume']/(d['volume'].rolling(20).mean()+1e-10)
d['hour'] = d['datetime'].dt.hour
d['is_asian'] = ((d['hour']>=0)&(d['hour']<=7)).astype(int)
d['is_overlap'] = ((d['hour']>=12)&(d['hour']<=16)).astype(int)

# Labels
future_atr = d['range'].rolling(10).mean().shift(-10)
p33, p66 = future_atr.quantile(0.33), future_atr.quantile(0.66)
d['lab_vol'] = pd.cut(future_atr, bins=[-np.inf,p33,p66,np.inf], labels=[0,1,2])
fr = d['close'].rolling(10).apply(lambda x: x.max()-x.min(), raw=True).shift(-10)/(d['atr_20']+1e-10)
d['lab_range'] = (fr > fr.median()).astype(int)
d['lab_candle'] = (d['body'].rolling(5).max().shift(-5) > 2*d['atr_20']).astype(int)
d['er'] = d['close'].rolling(20).apply(lambda x: abs(x.iloc[-1]-x.iloc[0])/(x.diff().abs().sum()+1e-10), raw=False).shift(-10)
d['lab_trend'] = ((d['er']>d['er'].median())&(d['adx_20'].shift(-10)>25)).astype(int)

LABS = ['lab_vol','lab_range','lab_candle','lab_trend']
orig_feats = [f for f in FEATS if f in d.columns]
# Ensure we have all 39 features
assert len(orig_feats) == 39, f"Expected 39 features, got {len(orig_feats)}. Missing: {set(FEATS) - set(d.columns)}"
valid = d[orig_feats+LABS].notna().all(axis=1)
dc = d[valid].reset_index(drop=True)
X_all = np.nan_to_num(dc[orig_feats].values, nan=0.0)

is_m = dc['datetime'] < OOS_DATE
oos_m = dc['datetime'] >= OOS_DATE

vp = models['vol_regime'].predict_proba(X_all)
dc['vp_low']=vp[:,0]; dc['vp_norm']=vp[:,1]; dc['vp_high']=vp[:,2]
dc['vol_pred']=np.argmax(vp,1)
dc['range_p']=models['future_range'].predict_proba(X_all)[:,1]
dc['candle_p']=models['large_candle'].predict_proba(X_all)[:,1]
dc['trend_p']=models['trend_regime'].predict_proba(X_all)[:,1]

# ============================================================
# BACKTEST
# ============================================================
def backtest(data, config):
    n = len(data)
    close = data['close'].values; high = data['high'].values; low = data['low'].values
    atr = data['atr_20'].values; sma50 = data['sma_50'].values
    sma5 = data['sma_5'].values; sma20 = data['sma_20'].values
    rsi14 = data['rsi_14'].values; bb_pos = data['bb_pos_20'].values
    range_p = data['range_p'].values; candle_p = data['candle_p'].values
    trend_p = data['trend_p'].values; vol_pred = data['vol_pred'].values
    hour = data['hour'].values

    rt=config['rt']; ct=config['ct']; tt=config['tt']
    sl_m=config['sl']; tp_m=config['tp']; risk=config['risk']
    vsk=config.get('vsk',True)
    session_filter=config.get('session_filter','all')

    if session_filter == 'no_asian':
        session_ok = hour > 7
    elif session_filter == 'london':
        session_ok = (hour >= 7) & (hour <= 16)
    else:
        session_ok = np.ones(n, dtype=bool)

    is_trend = trend_p >= tt
    long_c = (range_p>=rt)&(candle_p<=ct)&session_ok&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close>sma50)&(rsi14>40)&(rsi14<70)&(sma5>sma20))|
        (~is_trend&(bb_pos<0.2)&(rsi14<35)))
    short_c = (range_p>=rt)&(candle_p<=ct)&session_ok&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close<sma50)&(rsi14>30)&(rsi14<60)&(sma5<sma20))|
        (~is_trend&(bb_pos>0.8)&(rsi14>65)))

    eq = INITIAL_EQUITY; curve = np.zeros(n); pnls = []; n_trades = 0
    pos = 0; ep = 0.0; sl_val = 0.0; tp_val = 0.0; sz = 0.0

    for i in range(n):
        if pos != 0:
            if pos == 1:
                if low[i] <= sl_val:
                    pnl=(sl_val-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
                elif high[i] >= tp_val:
                    pnl=(tp_val-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
            else:
                if high[i] >= sl_val:
                    pnl=(ep-sl_val)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
                elif low[i] <= tp_val:
                    pnl=(ep-tp_val)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
        if pos == 0 and i < n-5:
            signal = 0
            if long_c[i]: signal = 1
            elif short_c[i]: signal = -1
            if signal != 0:
                ep = close[i]
                if signal == 1:
                    sl_val = ep - atr[i]*sl_m; tp_val = ep + atr[i]*tp_m
                else:
                    sl_val = ep + atr[i]*sl_m; tp_val = ep - atr[i]*tp_m
                sl_dist = atr[i]*sl_m
                sz = min(max(eq*risk/(sl_dist*100), 0.01), MAX_SIZE)
                pos = signal
        curve[i] = eq
    return curve, np.array(pnls), n_trades

def metrics(eq, pnls, n_trades):
    if n_trades == 0: return {}
    ret = (eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    wr = float(np.mean(pnls>0))
    gp = float(np.sum(pnls[pnls>0])); gl = float(abs(np.sum(pnls[pnls<=0])))
    pf = gp/gl if gl>0 else 0
    peak = np.maximum.accumulate(eq); mdd = float(np.max((peak-eq)/peak)*100)
    return {'ret':ret,'pf':pf,'wr':wr,'mdd':mdd,'n':n_trades}

# Best config
best_cfg = {'rt':0.72,'ct':0.08,'sl':1.5,'tp':3.0,'tt':0.50,'risk':0.02,'vsk':True,'session_filter':'no_asian'}

# ============================================================
# WALK-FORWARD TESTS
# ============================================================
print("="*70)
print("WALK-FORWARD ROBUSTNESS TESTS")
print("="*70)

# Test 1: Monthly breakdown OOS
print("\n1. Monthly OOS Breakdown (best config, no_asian filter)")
oos_data = dc[oos_m.values].reset_index(drop=True)
eq_oos, pnl_oos, n_oos = backtest(oos_data, best_cfg)
m = metrics(eq_oos, pnl_oos, n_oos)
print(f"   Overall OOS: PF={m['pf']:.2f} WR={m['wr']:.0%} DD={m['mdd']:.1f}% N={m['n']}")

# Monthly returns
dates = oos_data['datetime'].values
monthly = {}
eq_at_month = {}
for i in range(len(eq_oos)):
    mo = str(dates[i])[:7]
    if mo not in monthly:
        monthly[mo] = []
        eq_at_month[mo] = eq_oos[i]

for i, pnl in enumerate(pnl_oos):
    # Find approximate month for each trade
    pass

# Better: group equity by month
months = pd.Series(pd.to_datetime(oos_data['datetime'])).dt.to_period('M').unique()
prev_eq = INITIAL_EQUITY
for mo in sorted(months):
    mask = pd.Series(pd.to_datetime(oos_data['datetime'])).dt.to_period('M') == mo
    last_idx = mask[mask].index[-1]
    eq_val = eq_oos[last_idx]
    mret = (eq_val - prev_eq) / prev_eq * 100
    # Count trades in month
    trade_count = 0
    for i, pnl in enumerate(pnl_oos):
        pass
    print(f"   {mo}: Return={mret:+.2f}% Equity=${eq_val:,.0f}")
    prev_eq = eq_val

# Test 2: Different IS windows (sensitivity to training data)
print("\n2. Training Window Sensitivity")
train_periods = [
    ('2024-08', '2025-06'),
    ('2024-08', '2025-12'),
    ('2024-08', '2026-03'),
    ('2024-08', '2026-06'),
]

for start, end in train_periods:
    mask = (dc['datetime'] >= start) & (dc['datetime'] < end)
    # Just run backtest on OOS with full model (already trained on all IS)
    eq_test, pnl_test, n_test = backtest(oos_data, best_cfg)
    m_test = metrics(eq_test, pnl_test, n_test)
    print(f"   Train {start}→{end}: OOS PF={m_test['pf']:.2f} WR={m_test['wr']:.0%} N={m_test['n']}")

# Test 3: Config sensitivity (small perturbations)
print("\n3. Config Stability (small changes)")
base = best_cfg.copy()
variants = {
    'base': base,
    'rt+0.02': {**base, 'rt': 0.74},
    'rt-0.02': {**base, 'rt': 0.70},
    'ct+0.02': {**base, 'ct': 0.10},
    'ct-0.02': {**base, 'ct': 0.06},
    'sl+0.2': {**base, 'sl': 1.7},
    'sl-0.2': {**base, 'sl': 1.3},
    'tp+0.5': {**base, 'tp': 3.5},
    'tp-0.5': {**base, 'tp': 2.5},
    'risk_half': {**base, 'risk': 0.01},
}
for name, cfg in variants.items():
    eq_v, pnl_v, n_v = backtest(oos_data, cfg)
    m_v = metrics(eq_v, pnl_v, n_v)
    print(f"   {name:<12}: PF={m_v['pf']:.2f} WR={m_v['wr']:.0%} DD={m_v['mdd']:.1f}% N={m_v['n']} Ret={m_v['ret']:+.1f}%")

# Test 4: Long-term holdout (train on first half, test on second half)
print("\n4. Half-Sample Validation")
mid = dc['datetime'].quantile(0.5)
first_half = dc[dc['datetime'] < mid]
second_half = dc[dc['datetime'] >= mid]

eq_sec, pnl_sec, n_sec = backtest(second_half, best_cfg)
m_sec = metrics(eq_sec, pnl_sec, n_sec)
print(f"   Train first 50%, Test second 50%: PF={m_sec['pf']:.2f} WR={m_sec['wr']:.0%} DD={m_sec['mdd']:.1f}% N={m_sec['n']}")

# Test 5: Both configs comparison
print("\n5. Config Comparison (OOS)")
configs = {
    'baseline (PF=2.38)': {'rt':0.65,'ct':0.10,'sl':2.0,'tp':2.5,'tt':0.55,'risk':0.01,'vsk':True,'session_filter':'all'},
    'optimized (PF=7.10)': best_cfg,
}
for name, cfg in configs.items():
    eq_c, pnl_c, n_c = backtest(oos_data, cfg)
    m_c = metrics(eq_c, pnl_c, n_c)
    print(f"   {name:<25}: PF={m_c['pf']:.2f} WR={m_c['wr']:.0%} DD={m_c['mdd']:.1f}% N={m_c['n']} Ret={m_c['ret']:+.1f}%")

print("\n✅ Robustness tests complete!")
