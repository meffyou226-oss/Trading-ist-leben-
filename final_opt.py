#!/usr/bin/env python3
"""
Final optimization: Best of both worlds.
Use C3 config (PF=4.76, WR=76%) as base, add re-entry + secondary for frequency.
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

print(f"Ready: {len(dc)} bars")

# ============================================================
# BACKTEST WITH RE-ENTRY
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
    reentry_wait = config.get('reentry_wait', 0)  # bars to wait before re-entry

    sf = config.get('session_filter','all')
    if sf == 'no_asian': session_ok = hour > 7
    elif sf == 'london': session_ok = (hour >= 7) & (hour <= 16)
    else: session_ok = np.ones(n, dtype=bool)

    is_trend = trend_p >= tt
    long_c = (range_p>=rt)&(candle_p<=ct)&session_ok&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close>sma50)&(rsi14>40)&(rsi14<70)&(sma5>sma20))|
        (~is_trend&(bb_pos<0.2)&(rsi14<35)))
    short_c = (range_p>=rt)&(candle_p<=ct)&session_ok&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close<sma50)&(rsi14>30)&(rsi14<60)&(sma5<sma20))|
        (~is_trend&(bb_pos>0.8)&(rsi14>65)))

    eq = INITIAL_EQUITY; curve = np.zeros(n); pnls = []; n_trades = 0
    pos = 0; ep = 0.0; sl_val = 0.0; tp_val = 0.0; sz = 0.0
    last_exit = -10

    for i in range(n):
        if pos != 0:
            if pos == 1:
                if low[i] <= sl_val:
                    pnl=(sl_val-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0; last_exit=i
                elif high[i] >= tp_val:
                    pnl=(tp_val-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0; last_exit=i
            else:
                if high[i] >= sl_val:
                    pnl=(ep-sl_val)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0; last_exit=i
                elif low[i] <= tp_val:
                    pnl=(ep-tp_val)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0; last_exit=i

        if pos == 0 and (i - last_exit) > reentry_wait:
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

def metrics(eq, pnls, n_trades, trading_days=55):
    if n_trades == 0: return {}
    ret = (eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    wr = float(np.mean(pnls>0))
    gp = float(np.sum(pnls[pnls>0])); gl = float(abs(np.sum(pnls[pnls<=0])))
    pf = gp/gl if gl>0 else 0
    peak = np.maximum.accumulate(eq); mdd = float(np.max((peak-eq)/peak)*100)
    return {'ret':ret,'pf':pf,'wr':wr,'mdd':mdd,'n':n_trades,'tpd':n_trades/trading_days}

# ============================================================
# FINAL EXPERIMENTS
# ============================================================
is_data = dc[is_m.values].reset_index(drop=True)
oos_data = dc[oos_m.values].reset_index(drop=True)

# Best configs from previous experiments
base_pf7 = {'rt':0.72,'ct':0.08,'sl':1.5,'tp':3.0,'tt':0.50,'risk':0.02,'vsk':True,'session_filter':'no_asian'}
c3_pf476 = {'rt':0.72,'ct':0.08,'sl':1.2,'tp':2.5,'tt':0.50,'risk':0.02,'vsk':True,'session_filter':'no_asian'}

print("\n" + "="*70)
print("FINAL OPTIMIZATION: High PF + Frequency")
print("="*70)

experiments = {
    # Baseline PF=7
    '01_pf7_baseline': base_pf7,

    # PF=7 with re-entry (no wait)
    '02_pf7_reentry_0': {**base_pf7, 'reentry_wait':0},
    '03_pf7_reentry_5': {**base_pf7, 'reentry_wait':5},
    '04_pf7_reentry_10': {**base_pf7, 'reentry_wait':10},

    # C3 (PF=4.76) with re-entry
    '05_c3_reentry_0': {**c3_pf476, 'reentry_wait':0},
    '06_c3_reentry_5': {**c3_pf476, 'reentry_wait':5},
    '07_c3_reentry_10': {**c3_pf476, 'reentry_wait':10},

    # C3 with tighter SL for faster trades
    '08_c3_sl10_re0': {**c3_pf476, 'sl':1.0, 'reentry_wait':0},
    '09_c3_sl10_re5': {**c3_pf476, 'sl':1.0, 'reentry_wait':5},

    # C3 with lower RT for more trades
    '10_c3_rt70_re0': {**c3_pf476, 'rt':0.70, 'reentry_wait':0},
    '11_c3_rt68_re0': {**c3_pf476, 'rt':0.68, 'reentry_wait':0},
    '12_c3_rt65_re0': {**c3_pf476, 'rt':0.65, 'reentry_wait':0},

    # C3 with higher risk
    '13_c3_risk3_re0': {**c3_pf476, 'risk':0.03, 'reentry_wait':0},
    '14_c3_risk4_re0': {**c3_pf476, 'risk':0.04, 'reentry_wait':0},

    # Best from frequency opt (PF=2.79)
    '15_freq_best': {'rt':0.65,'ct':0.10,'sl':1.0,'tp':2.5,'tt':0.50,'risk':0.01,'vsk':True,'session_filter':'no_asian','reentry_wait':0},

    # Hybrid: C3 + freq params
    '16_c3_rt65_sl12_re0': {**c3_pf476, 'rt':0.65, 'sl':1.2, 'reentry_wait':0},
    '17_c3_rt68_sl12_re0': {**c3_pf476, 'rt':0.68, 'sl':1.2, 'reentry_wait':0},
    '18_c3_rt65_sl10_re0': {**c3_pf476, 'rt':0.65, 'sl':1.0, 'reentry_wait':0},
    '19_c3_rt68_sl10_tp25_re0': {**c3_pf476, 'rt':0.68, 'sl':1.0, 'tp':2.5, 'reentry_wait':0},
    '20_c3_rt70_sl12_tp22_re0': {**c3_pf476, 'rt':0.70, 'sl':1.2, 'tp':2.2, 'reentry_wait':0},
}

print(f"\n{'#':<4} {'Config':<28} {'IS PF':>6} {'T/D':>5} {'OOS PF':>6} {'WR':>5} {'DD%':>5} {'T/D':>5} {'N':>5} {'Ret%':>8}")
print("─"*90)

results = []
for name, cfg in experiments.items():
    try:
        eq_is, pnl_is, n_is = backtest(is_data, cfg)
        eq_oos, pnl_oos, n_oos = backtest(oos_data, cfg)
        m_is = metrics(eq_is, pnl_is, n_is, 550)
        m_oos = metrics(eq_oos, pnl_oos, n_oos, 55)
        if not m_oos or m_oos['n'] < 10: continue
        results.append({'name':name,'cfg':cfg,'is':m_is,'oos':m_oos})

        in_range = "  ✅" if 2 <= m_oos['tpd'] <= 10 else ""
        print(f"{len(results):<4} {name:<28} {m_is['pf']:>6.2f} {m_is['tpd']:>5.1f} "
              f"{m_oos['pf']:>6.2f} {m_oos['wr']:>4.0%} {m_oos['mdd']:>4.1f} "
              f"{m_oos['tpd']:>5.1f} {m_oos['n']:>5} {m_oos['ret']:>+7.1f}%{in_range}")
    except Exception as e:
        print(f"  Error {name}: {e}")

# Top by PF in range
print("\n" + "="*70)
print("TOP BY OOS PF (2-10 trades/day)")
print("="*70)
in_range = [r for r in results if 2 <= r['oos']['tpd'] <= 10 and r['oos']['pf'] >= 2.0]
in_range.sort(key=lambda x: x['oos']['pf'], reverse=True)

for i, r in enumerate(in_range[:10]):
    oos = r['oos']
    print(f"  {i+1}. {r['name']:<28} PF={oos['pf']:.2f} WR={oos['wr']:.0%} DD={oos['mdd']:.1f}% "
          f"T/D={oos['tpd']:.1f} N={oos['n']} Ret={oos['ret']:+.1f}%")

# Best overall (PF > 3, T/D > 2)
print("\n" + "="*70)
print("BEST: PF > 3.0 + 2-10 trades/day")
print("="*70)
high_pf = [r for r in results if 2 <= r['oos']['tpd'] <= 10 and r['oos']['pf'] >= 3.0]
high_pf.sort(key=lambda x: x['oos']['pf'], reverse=True)

for i, r in enumerate(high_pf[:5]):
    oos = r['oos']
    print(f"  {i+1}. {r['name']:<28} PF={oos['pf']:.2f} WR={oos['wr']:.0%} DD={oos['mdd']:.1f}% "
          f"T/D={oos['tpd']:.1f} N={oos['n']} Ret={oos['ret']:+.1f}%")
    print(f"      Config: {json.dumps(r['cfg'])}")

# Save best
if high_pf:
    best = high_pf[0]
    with open('best_final.json', 'w') as f:
        json.dump(best, f, indent=2, default=str)
    print(f"\nSaved best_final.json")
elif in_range:
    best = in_range[0]
    with open('best_final.json', 'w') as f:
        json.dump(best, f, indent=2, default=str)
    print(f"\nSaved best_final.json (PF={best['oos']['pf']:.2f})")

print("\n✅ Final optimization complete!")
