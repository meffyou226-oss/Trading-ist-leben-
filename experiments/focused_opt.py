#!/usr/bin/env python3
"""
Focused Optimization around best findings:
- RT=0.70 gave PF=3.77 (highest OOS)
- TP=3.5 gave PF=3.04
- SL=1.5 was best
Explore combinations around these values.
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
d['is_london'] = ((d['hour']>=7)&(d['hour']<=16)).astype(int)
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

print(f"Ready: {len(dc)} bars")

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
    use_trailing=config.get('trailing',False)
    trail_atr=config.get('trail_atr',1.5)
    session_filter=config.get('session_filter','all')

    if session_filter == 'london':
        session_ok = (hour >= 7) & (hour <= 16)
    elif session_filter == 'overlap':
        session_ok = (hour >= 12) & (hour <= 16)
    elif session_filter == 'no_asian':
        session_ok = hour > 7
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
        if use_trailing and pos != 0:
            if pos == 1:
                new_sl = close[i] - trail_atr * atr[i]
                if new_sl > sl_val: sl_val = new_sl
            else:
                new_sl = close[i] + trail_atr * atr[i]
                if new_sl < sl_val: sl_val = new_sl

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
    wr = np.mean(pnls>0)
    gp = np.sum(pnls[pnls>0]); gl = abs(np.sum(pnls[pnls<=0]))
    pf = gp/gl if gl>0 else 0
    peak = np.maximum.accumulate(eq); mdd = np.max((peak-eq)/peak)*100
    rets = np.diff(eq)/eq[:-1]; rets=rets[rets!=0]
    shr = np.mean(rets)/np.std(rets)*np.sqrt(252*288) if len(rets)>1 else 0
    return {'ret':ret,'pf':pf,'wr':wr,'mdd':mdd,'n':n_trades,'sharpe':shr}

# ============================================================
# FOCUSED OPTIMIZATION
# ============================================================
is_data = dc[is_m.values].reset_index(drop=True)
oos_data = dc[oos_m.values].reset_index(drop=True)

print("\n" + "="*70)
print("FOCUSED OPTIMIZATION (RT=0.60-0.75, TP=2.5-4.0)")
print("="*70)

from itertools import product
grid = {
    'rt': [0.60, 0.65, 0.68, 0.70, 0.72, 0.75],
    'ct': [0.08, 0.10, 0.12, 0.15],
    'sl': [1.2, 1.5, 1.8, 2.0],
    'tp': [2.5, 3.0, 3.5, 4.0],
    'tt': [0.50, 0.55],
    'risk': [0.01, 0.015, 0.02],
    'vsk': [True],
    'trailing': [False],
    'session_filter': ['all'],
}

keys = list(grid.keys())
combos = list(product(*grid.values()))
print(f"Testing {len(combos)} combinations...")

best_score = -999
best_config = None
all_results = []

for combo in combos:
    cfg = dict(zip(keys, combo))
    try:
        eq_is, pnl_is, n_is = backtest(is_data, cfg)
        eq_oos, pnl_oos, n_oos = backtest(oos_data, cfg)
        m_is = metrics(eq_is, pnl_is, n_is)
        m_oos = metrics(eq_oos, pnl_oos, n_oos)
        if n_oos < 20 or n_is < 30: continue

        # Score: OIS PF penalized by DD
        score = m_oos['pf'] * 10 - m_oos['mdd'] + m_oos['wr'] * 5
        all_results.append({'cfg':cfg,'is':m_is,'oos':m_oos,'score':score})
        if score > best_score:
            best_score = score
            best_config = cfg
    except: continue

all_results.sort(key=lambda x: x['score'], reverse=True)

print(f"\nTop 15 configurations (OOS):")
print(f"{'#':<3} {'RT':>4} {'CT':>4} {'SL':>4} {'TP':>4} {'TT':>4} {'Rsk':>4} | {'OOS PF':>6} {'WR':>5} {'DD%':>5} {'N':>4} {'Ret%':>7} | {'IS PF':>6} {'N':>4}")
for i, r in enumerate(all_results[:15]):
    c = r['cfg']
    oos = r['oos']; ism = r['is']
    print(f"{i+1:<3} {c['rt']:>4.2f} {c['ct']:>4.2f} {c['sl']:>4.1f} {c['tp']:>4.1f} {c['tt']:>4.2f} {c['risk']:>4.2f} | "
          f"{oos['pf']:>6.2f} {oos['wr']:>4.0%} {oos['mdd']:>4.1f} {oos['n']:>4} {oos['ret']:>+6.1f}% | "
          f"{ism['pf']:>6.2f} {ism['n']:>4}")

# Best config
bc = all_results[0]['cfg']
print(f"\nBest config: {json.dumps(bc)}")
print(f"OOS: PF={all_results[0]['oos']['pf']:.2f} WR={all_results[0]['oos']['wr']:.0%} DD={all_results[0]['oos']['mdd']:.1f}% N={all_results[0]['oos']['n']}")

# ============================================================
# TEST BEST WITH TRAILING STOP
# ============================================================
print("\n" + "="*70)
print("TEST BEST CONFIG + TRAILING STOP")
print("="*70)

for trail in [False, 1.5, 2.0, 2.5]:
    cfg = bc.copy()
    cfg['trailing'] = trail > 0
    cfg['trail_atr'] = trail
    eq_oos, pnl_oos, n_oos = backtest(oos_data, cfg)
    m = metrics(eq_oos, pnl_oos, n_oos)
    if m:
        print(f"  Trail={trail}: PF={m['pf']:.2f} WR={m['wr']:.0%} DD={m['mdd']:.1f}% N={m['n']} Ret={m['ret']:+.1f}%")

# ============================================================
# TEST BEST WITH SESSION FILTERS
# ============================================================
print("\n" + "="*70)
print("TEST BEST CONFIG + SESSION FILTERS")
print("="*70)

for sf in ['all', 'london', 'overlap', 'no_asian']:
    cfg = bc.copy()
    cfg['session_filter'] = sf
    cfg['trailing'] = False
    eq_oos, pnl_oos, n_oos = backtest(oos_data, cfg)
    m = metrics(eq_oos, pnl_oos, n_oos)
    if m:
        print(f"  Session={sf:<10}: PF={m['pf']:.2f} WR={m['wr']:.0%} DD={m['mdd']:.1f}% N={m['n']} Ret={m['ret']:+.1f}%")

# ============================================================
# FINAL BEST
# ============================================================
print("\n" + "="*70)
print("FINAL BEST CONFIG")
print("="*70)

# Test top 3 with trailing + session
final_tests = []
for r in all_results[:5]:
    for trail in [False, 2.0]:
        for sf in ['all', 'no_asian']:
            cfg = r['cfg'].copy()
            cfg['trailing'] = trail > 0
            cfg['trail_atr'] = trail
            cfg['session_filter'] = sf
            eq_oos, pnl_oos, n_oos = backtest(oos_data, cfg)
            m = metrics(eq_oos, pnl_oos, n_oos)
            if m and n_oos > 15:
                final_tests.append({'cfg':cfg,**m,'trail':trail,'sf':sf})

final_tests.sort(key=lambda x: x['pf']*10 - x['mdd'], reverse=True)

print(f"\nTop 10 final (trailing x session):")
for i, r in enumerate(final_tests[:10]):
    print(f"  {i+1}. PF={r['pf']:.2f} WR={r['wr']:.0%} DD={r['mdd']:.1f}% N={r['n']} Ret={r['ret']:+.1f}% | "
          f"RT={r['cfg']['rt']} CT={r['cfg']['ct']} SL={r['cfg']['sl']} TP={r['cfg']['tp']} Trail={r['trail']} SF={r['sf']}")

# Save best
best_final = final_tests[0]
with open('best_config.json', 'w') as f:
    json.dump(best_final, f, indent=2, default=str)
print(f"\nSaved best_config.json")
print(f"\n✅ Focused optimization complete!")
