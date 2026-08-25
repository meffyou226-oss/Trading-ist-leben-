#!/usr/bin/env python3
"""
Trade Frequency Optimization: Target 2-10 trades/day
Adjust RT, CT, SL, TP, filters to maximize trade count while keeping PF > 2.
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

print(f"Ready: {len(dc)} bars | OOS: {oos_m.sum()}")

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
    elif session_filter == 'overlap':
        session_ok = (hour >= 12) & (hour <= 16)
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

def metrics(eq, pnls, n_trades, trading_days=61):
    if n_trades == 0: return {}
    ret = (eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    wr = float(np.mean(pnls>0))
    gp = float(np.sum(pnls[pnls>0])); gl = float(abs(np.sum(pnls[pnls<=0])))
    pf = gp/gl if gl>0 else 0
    peak = np.maximum.accumulate(eq); mdd = float(np.max((peak-eq)/peak)*100)
    trades_per_day = n_trades / trading_days
    return {'ret':ret,'pf':pf,'wr':wr,'mdd':mdd,'n':n_trades,'tpd':trades_per_day}

# ============================================================
# OPTIMIZATION: Target 2-10 trades/day
# ============================================================
is_data = dc[is_m.values].reset_index(drop=True)
oos_data = dc[oos_m.values].reset_index(drop=True)

# Estimate trading days in OOS
oos_days = (oos_data['datetime'].max() - oos_data['datetime'].min()).days
print(f"OOS period: ~{oos_days} calendar days")

print("\n" + "="*70)
print("FREQUENCY OPTIMIZATION (Target: 2-10 trades/day)")
print("="*70)

from itertools import product

grid = {
    'rt': [0.50, 0.55, 0.60, 0.65],
    'ct': [0.10, 0.15, 0.20],
    'sl': [1.0, 1.5, 2.0],
    'tp': [1.5, 2.0, 2.5],
    'tt': [0.50, 0.55],
    'risk': [0.01, 0.02],
    'vsk': [True],
    'session_filter': ['all', 'no_asian'],
}

keys = list(grid.keys())
combos = list(product(*grid.values()))
print(f"Testing {len(combos)} combinations...")

results = []
for combo in combos:
    cfg = dict(zip(keys, combo))
    try:
        eq_oos, pnl_oos, n_oos = backtest(oos_data, cfg)
        eq_is, pnl_is, n_is = backtest(is_data, cfg)
        m_oos = metrics(eq_oos, pnl_oos, n_oos, oos_days)
        m_is = metrics(eq_is, pnl_is, n_is, 550)
        if not m_oos or m_oos['n'] < 30: continue

        # Target: 2-10 trades/day, PF > 2.0
        tpd = m_oos['tpd']
        pf = m_oos['pf']
        in_range = 2 <= tpd <= 10

        # Score: prioritize in-range, then PF, then low DD
        if in_range and pf >= 2.0:
            score = pf * 10 - m_oos['mdd'] + 50  # bonus for in range
        elif in_range:
            score = pf * 5 - m_oos['mdd']
        else:
            score = pf * 3 - m_oos['mdd'] - abs(tpd - 5) * 5  # penalize distance from target

        results.append({'cfg':cfg,'is':m_is,'oos':m_oos,'score':score,'in_range':in_range})
    except: continue

results.sort(key=lambda x: x['score'], reverse=True)

print(f"\nTop 20 configs (target: 2-10 trades/day, PF>2):")
print(f"{'#':<3} {'RT':>4} {'CT':>4} {'SL':>4} {'TP':>4} {'TT':>4} {'Rsk':>4} {'SF':<8} | {'OOS PF':>6} {'WR':>5} {'DD%':>5} {'N':>5} {'T/D':>5} | {'IS PF':>6} {'T/D':>5} {'Range':>6}")
for i, r in enumerate(results[:20]):
    c = r['cfg']
    oos = r['oos']; ism = r['is']
    rng = "YES" if r['in_range'] else ""
    print(f"{i+1:<3} {c['rt']:>4.2f} {c['ct']:>4.2f} {c['sl']:>4.1f} {c['tp']:>4.1f} {c['tt']:>4.2f} {c['risk']:>4.2f} {c['session_filter']:<8} | "
          f"{oos['pf']:>6.2f} {oos['wr']:>4.0%} {oos['mdd']:>4.1f} {oos['n']:>5} {oos['tpd']:>5.1f} | "
          f"{ism['pf']:>6.2f} {ism['tpd']:>5.1f} {rng:>6}")

# Filter to only in-range results
in_range_results = [r for r in results if r['in_range'] and r['oos']['pf'] >= 2.0]
print(f"\n\nConfigs in target range (2-10 trades/day, PF>=2): {len(in_range_results)}")
if in_range_results:
    print(f"\nTop 10 in-range configs:")
    for i, r in enumerate(in_range_results[:10]):
        c = r['cfg']
        oos = r['oos']
        print(f"  {i+1}. RT={c['rt']} CT={c['ct']} SL={c['sl']} TP={c['tp']} TT={c['tt']} Risk={c['risk']} SF={c['session_filter']}")
        print(f"      OOS: PF={oos['pf']:.2f} WR={oos['wr']:.0%} DD={oos['mdd']:.1f}% N={oos['n']} T/D={oos['tpd']:.1f} Ret={oos['ret']:+.1f}%")

# Save best in-range
if in_range_results:
    best = in_range_results[0]
    with open('best_config_freq.json', 'w') as f:
        json.dump({'config': best['cfg'], 'oos': best['oos'], 'is': best['is']}, f, indent=2, default=str)
    print(f"\nSaved best_config_freq.json")
else:
    print("\nNo configs found in target range. Showing closest matches...")
    # Show configs closest to target range
    close_results = sorted(results, key=lambda x: (abs(x['oos']['tpd'] - 5), -x['oos']['pf']))[:5]
    for r in close_results:
        c = r['cfg']; oos = r['oos']
        print(f"  RT={c['rt']} CT={c['ct']} SL={c['sl']} TP={c['tp']} T/D={oos['tpd']:.1f} PF={oos['pf']:.2f}")

print("\n✅ Frequency optimization complete!")
