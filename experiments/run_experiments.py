#!/usr/bin/env python3
"""
Advanced Strategy Optimization - Local Experiments
Tests: Trailing Stop, Dynamic Sizing, Session Filters, More Features
NO changes to main repo - all local in experiments/
"""
import pandas as pd, numpy as np, xgboost as xgb, pickle, json, os, sys
from sklearn.metrics import roc_auc_score
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

# ============================================================
# 1. LOAD DATA, MODELS, EXTENDED FEATURES
# ============================================================
print("="*70)
print("LOADING DATA & MODELS")
print("="*70)

DATA_FILES = [f"XAUUSD_M5_{y}-{m:02d}.csv" for y in range(2024,2027) for m in range(1,13)
              if (y==2024 and m>=8) or (y==2025) or (y==2026 and m<=8)]

dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in DATA_FILES]
df = pd.concat(dfs, ignore_index=True)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values('datetime').reset_index(drop=True)

# Load saved models
models = {}
for name in ['vol_regime', 'future_range', 'large_candle', 'trend_regime']:
    with open(os.path.join(MODELS_DIR, f'{name}.pkl'), 'rb') as f:
        models[name] = pickle.load(f)
print("Loaded 4 models")

with open(os.path.join(CONFIG_DIR, 'features.json'), 'r') as f:
    FEATS = json.load(f)

# Extended feature engineering
print("Engineering extended features...")
d = df.copy()
d['returns'] = d['close'].pct_change()
d['log_ret'] = np.log(d['close']/d['close'].shift(1))
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
    d[f'plus_di_{p}'] = pdi; d[f'minus_di_{p}'] = mdi
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
d['minute'] = d['datetime'].dt.minute
d['day_of_week'] = d['datetime'].dt.dayofweek
d['is_asian'] = ((d['hour']>=0)&(d['hour']<=7)).astype(int)
d['is_london'] = ((d['hour']>=7)&(d['hour']<=16)).astype(int)
d['is_ny'] = ((d['hour']>=12)&(d['hour']<=21)).astype(int)
d['is_overlap'] = ((d['hour']>=12)&(d['hour']<=16)).astype(int)
d['is_us_open'] = ((d['hour']==12)&(d['minute']<=30)).astype(int)
d['is_eu_open'] = ((d['hour']==7)&(d['minute']<=30)).astype(int)

# Extended features
d['rsi_divergence'] = d['rsi_14'] - d['rsi_14'].shift(5)
d['rsi_slope'] = d['rsi_14'].diff(3)
d['macd_cross'] = np.sign(d['close'].ewm(12).mean()-d['close'].ewm(26).mean())
d['consec_bars_up'] = d['returns'].groupby((d['returns']<=0).cumsum()).cumcount()
d['consec_bars_dn'] = d['returns'].groupby((d['returns']>=0).cumsum()).cumcount()
d['price_accel'] = d['returns'].diff(3)
d['vol_expanding'] = (d['range'] > d['range'].rolling(10).mean()).astype(int)
d['vol_contracting'] = (d['range'] < d['range'].rolling(10).mean()*0.5).astype(int)
d['inside_bar'] = ((d['high']<d['high'].shift(1))&(d['low']>d['low'].shift(1))).astype(int)
d['outside_bar'] = ((d['high']>d['high'].shift(1))&(d['low']<d['low'].shift(1))).astype(int)
d['gap'] = d['open'] - d['close'].shift(1)
d['gap_ratio'] = d['gap']/(d['atr_20']+1e-10)

# Session volatility
d['session_vol'] = d.groupby('hour')['range'].transform(lambda x: x.rolling(20).mean())

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

# Generate predictions with saved models (use only original 39 features)
# Models were trained on original feature set from config
orig_feats = [f for f in FEATS if f in d.columns]
assert len(orig_feats) == 39, f"Expected 39 features, got {len(orig_feats)}"

valid = d[orig_feats+LABS].notna().all(axis=1)
dc = d[valid].reset_index(drop=True)
X_all = np.nan_to_num(dc[orig_feats].values, nan=0.0)

is_m = dc['datetime'] < OOS_DATE
oos_m = dc['datetime'] >= OOS_DATE

# Generate predictions (models trained on IS only)
vp = models['vol_regime'].predict_proba(X_all)
dc['vp_low']=vp[:,0]; dc['vp_norm']=vp[:,1]; dc['vp_high']=vp[:,2]
dc['vol_pred']=np.argmax(vp,1)
dc['range_p']=models['future_range'].predict_proba(X_all)[:,1]
dc['candle_p']=models['large_candle'].predict_proba(X_all)[:,1]
dc['trend_p']=models['trend_regime'].predict_proba(X_all)[:,1]

print(f"Ready: {len(dc)} bars | Features: {len(orig_feats)}")

# ============================================================
# 2. ADVANCED BACKTEST ENGINE
# ============================================================
def backtest_advanced(data, config):
    """Advanced backtest with trailing stop, dynamic sizing, session filter."""
    n = len(data)
    close = data['close'].values; high = data['high'].values; low = data['low'].values
    atr = data['atr_20'].values; sma50 = data['sma_50'].values
    sma5 = data['sma_5'].values; sma20 = data['sma_20'].values
    rsi14 = data['rsi_14'].values; bb_pos = data['bb_pos_20'].values
    adx20 = data['adx_20'].values if 'adx_20' in data.columns else np.full(n,20)
    plus_di20 = data['plus_di_20'].values if 'plus_di_20' in data.columns else np.zeros(n)
    minus_di20 = data['minus_di_20'].values if 'minus_di_20' in data.columns else np.zeros(n)
    range_p = data['range_p'].values; candle_p = data['candle_p'].values
    trend_p = data['trend_p'].values; vol_pred = data['vol_pred'].values
    hour = data['hour'].values; minute = data['minute'].values
    is_asian = data['is_asian'].values
    inside_bar = data['inside_bar'].values if 'inside_bar' in data.columns else np.zeros(n)
    vol_contracting = data['vol_contracting'].values if 'vol_contracting' in data.columns else np.zeros(n)

    # Config
    rt=config['rt']; ct=config['ct']; tt=config['tt']
    sl_m=config['sl']; tp_m=config['tp']; risk=config['risk']
    vsk=config.get('vsk',True)
    use_trailing=config.get('trailing',False)
    trail_atr=config.get('trail_atr',1.5)
    session_filter=config.get('session_filter','all')  # all, london, overlap, no_asian
    kelly_factor=config.get('kelly',0.0)  # 0=fixed fractional, >0=kelly sizing
    confidence_filter=config.get('confidence',0.0)  # min model confidence

    # Precompute session mask
    if session_filter == 'london':
        session_ok = (hour >= 7) & (hour <= 16)
    elif session_filter == 'overlap':
        session_ok = (hour >= 12) & (hour <= 16)
    elif session_filter == 'no_asian':
        session_ok = hour > 7
    else:
        session_ok = np.ones(n, dtype=bool)

    is_trend = trend_p >= tt
    # Confidence filter: require all models to agree (high confidence)
    confidence_ok = (range_p > 0.5 + confidence_filter/2) & (candle_p < 0.5 - confidence_filter/2)

    long_c = (range_p>=rt)&(candle_p<=ct)&confidence_ok&session_ok&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close>sma50)&(rsi14>40)&(rsi14<70)&(sma5>sma20))|
        (~is_trend&(bb_pos<0.2)&(rsi14<35)))
    short_c = (range_p>=rt)&(candle_p<=ct)&confidence_ok&session_ok&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close<sma50)&(rsi14>30)&(rsi14<60)&(sma5<sma20))|
        (~is_trend&(bb_pos>0.8)&(rsi14>65)))

    eq = INITIAL_EQUITY; curve = np.zeros(n); pnls = []; n_trades = 0
    wins = 0; losses = 0
    pos = 0; ep = 0.0; sl_val = 0.0; tp_val = 0.0; sz = 0.0
    entry_idx = 0

    for i in range(n):
        # Trailing stop logic
        if use_trailing and pos != 0:
            if pos == 1:  # Long
                new_sl = close[i] - trail_atr * atr[i]
                if new_sl > sl_val:
                    sl_val = new_sl
                if new_sl > ep:  # Lock in profit
                    sl_val = max(sl_val, new_sl)
            else:  # Short
                new_sl = close[i] + trail_atr * atr[i]
                if new_sl < sl_val:
                    sl_val = new_sl
                if new_sl < ep:
                    sl_val = min(sl_val, new_sl)

        # Check exits
        if pos != 0:
            if pos == 1:
                if low[i] <= sl_val:
                    pnl=(sl_val-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl)
                    if pnl > 0: wins+=1
                    else: losses+=1
                    n_trades+=1; pos=0
                elif high[i] >= tp_val:
                    pnl=(tp_val-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl)
                    if pnl > 0: wins+=1
                    else: losses+=1
                    n_trades+=1; pos=0
            else:
                if high[i] >= sl_val:
                    pnl=(ep-sl_val)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl)
                    if pnl > 0: wins+=1
                    else: losses+=1
                    n_trades+=1; pos=0
                elif low[i] <= tp_val:
                    pnl=(ep-tp_val)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl)
                    if pnl > 0: wins+=1
                    else: losses+=1
                    n_trades+=1; pos=0

        # Entry
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

                # Position sizing
                sl_dist = atr[i]*sl_m
                if kelly_factor > 0 and wins+losses > 20:
                    win_rate = wins/(wins+losses)
                    avg_win = np.mean([p for p in pnls if p>0]) if wins>0 else 0
                    avg_loss = abs(np.mean([p for p in pnls if p<=0])) if losses>0 else 1
                    kelly = win_rate - (1-win_rate)/(avg_win/avg_loss) if avg_loss>0 else 0
                    kelly = max(0, min(kelly*kelly_factor, 0.25))
                    sz = min(max(eq*kelly/(sl_dist*100), 0.01), MAX_SIZE)
                else:
                    sz = min(max(eq*risk/(sl_dist*100), 0.01), MAX_SIZE)

                pos = signal; entry_idx = i

        curve[i] = eq
    return curve, np.array(pnls), n_trades

def metrics(eq, pnls, n_trades, label):
    if n_trades == 0: return f"{label}: No trades"
    ret = (eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    wr = np.mean(pnls>0)
    gp = np.sum(pnls[pnls>0]); gl = abs(np.sum(pnls[pnls<=0]))
    pf = gp/gl if gl>0 else 0
    peak = np.maximum.accumulate(eq); mdd = np.max((peak-eq)/peak)*100
    rets = np.diff(eq)/eq[:-1]; rets=rets[rets!=0]
    shr = np.mean(rets)/np.std(rets)*np.sqrt(252*288) if len(rets)>1 else 0
    return {'label':label,'ret':ret,'pf':pf,'wr':wr,'mdd':mdd,'n':n_trades,'sharpe':shr,'eq':eq[-1]}

# ============================================================
# 3. RUN EXPERIMENTS
# ============================================================
is_data = dc[is_m.values].reset_index(drop=True)
oos_data = dc[oos_m.values].reset_index(drop=True)

base_config = {'rt':0.65,'ct':0.10,'tt':0.55,'sl':2.0,'tp':2.5,'risk':0.01,'vsk':True}

experiments = {
    # Baseline
    '01_baseline': base_config.copy(),

    # Trailing Stop
    '02_trailing_15': {**base_config, 'trailing':True, 'trail_atr':1.5},
    '03_trailing_20': {**base_config, 'trailing':True, 'trail_atr':2.0},
    '04_trailing_25_tp35': {**base_config, 'tp':3.0, 'trailing':True, 'trail_atr':2.5},

    # Session Filters
    '05_london_only': {**base_config, 'session_filter':'london'},
    '06_overlap_only': {**base_config, 'session_filter':'overlap'},
    '07_no_asian': {**base_config, 'session_filter':'no_asian'},

    # Confidence Filter
    '08_confidence_1': {**base_config, 'confidence':0.1},
    '09_confidence_2': {**base_config, 'confidence':0.2},
    '10_confidence_3': {**base_config, 'confidence':0.3},

    # Dynamic SL/TP
    '11_sl15_tp30': {**base_config, 'sl':1.5, 'tp':3.0},
    '12_sl25_tp35': {**base_config, 'sl':2.5, 'tp':3.5},
    '13_sl18_tp22': {**base_config, 'sl':1.8, 'tp':2.2},

    # Combined: Trailing + London + Confidence
    '14_combo_trail_london': {**base_config, 'trailing':True, 'trail_atr':2.0, 'session_filter':'london', 'confidence':0.1},
    '15_combo_trail_overlap': {**base_config, 'trailing':True, 'trail_atr':1.5, 'session_filter':'overlap', 'confidence':0.15},
    '16_combo_all': {**base_config, 'tp':3.0, 'trailing':True, 'trail_atr':2.0, 'session_filter':'no_asian', 'confidence':0.1},

    # Kelly Sizing
    '17_kelly_05': {**base_config, 'kelly':0.5},
    '18_kelly_1': {**base_config, 'kelly':1.0},
    '19_kelly_05_combo': {**base_config, 'kelly':0.5, 'trailing':True, 'trail_atr':2.0, 'session_filter':'no_asian'},
}

print("\n" + "="*70)
print("EXPERIMENT RESULTS")
print("="*70)
print(f"{'#':<4} {'Experiment':<28} {'IS':>30} {'OOS':>30}")
print(f"{'':4} {'':28} {'PF   WR   DD%  N':>30} {'PF   WR   DD%  N':>30}")
print("─"*95)

results = []
for name, config in experiments.items():
    eq_is, pnl_is, n_is = backtest_advanced(is_data, config)
    eq_oos, pnl_oos, n_oos = backtest_advanced(oos_data, config)
    m_is = metrics(eq_is, pnl_is, n_is, "IS")
    m_oos = metrics(eq_oos, pnl_oos, n_oos, "OOS")
    results.append({'name':name,'config':config,'is':m_is,'oos':m_oos})

    is_str = f"{m_is['pf']:.2f} {m_is['wr']:.0%} {m_is['mdd']:>4.1f} {m_is['n']:>4}" if m_is['n']>0 else "No trades"
    oos_str = f"{m_oos['pf']:.2f} {m_oos['wr']:.0%} {m_oos['mdd']:>4.1f} {m_oos['n']:>4}" if m_oos['n']>0 else "No trades"
    print(f"{len(results):<4} {name:<28} {is_str:>30} {oos_str:>30}")

# ============================================================
# 4. BEST CONFIG ANALYSIS
# ============================================================
print("\n" + "="*70)
print("TOP 10 BY OOS PROFIT FACTOR")
print("="*70)
sorted_by_pf = sorted([r for r in results if r['oos']['n']>20], key=lambda x: x['oos']['pf'], reverse=True)
for i, r in enumerate(sorted_by_pf[:10]):
    oos = r['oos']
    print(f"  {i+1}. {r['name']:<28} PF={oos['pf']:.2f} WR={oos['wr']:.0%} DD={oos['mdd']:.1f}% N={oos['n']} Ret={oos['ret']:+.1f}%")

print("\n" + "="*70)
print("TOP 10 BY OOS RETURN")
print("="*70)
sorted_by_ret = sorted([r for r in results if r['oos']['n']>20], key=lambda x: x['oos']['ret'], reverse=True)
for i, r in enumerate(sorted_by_ret[:10]):
    oos = r['oos']
    print(f"  {i+1}. {r['name']:<28} Ret={oos['ret']:+.1f}% PF={oos['pf']:.2f} DD={oos['mdd']:.1f}% N={oos['n']}")

# ============================================================
# 5. SENSITIVITY ANALYSIS (around best config)
# ============================================================
print("\n" + "="*70)
print("SENSITIVITY ANALYSIS (Best Config)")
print("="*70)

best = sorted_by_pf[0]
print(f"Best: {best['name']} | Config: {json.dumps(best['config'])}")

# Vary one parameter at a time
sens_params = ['rt', 'ct', 'sl', 'tp', 'risk']
for sp in sens_params:
    print(f"\n  Varying {sp}:")
    base_val = best['config'].get(sp, 0.65)
    if sp in ['rt', 'ct']:
        test_vals = [base_val-0.05, base_val, base_val+0.05]
    elif sp in ['sl', 'tp']:
        test_vals = [base_val-0.5, base_val, base_val+0.5]
    elif sp == 'risk':
        test_vals = [0.005, 0.01, 0.015, 0.02]
    else:
        test_vals = [base_val]

    for tv in test_vals:
        cfg = best['config'].copy()
        cfg[sp] = tv
        eq_oos, pnl_oos, n_oos = backtest_advanced(oos_data, cfg)
        m = metrics(eq_oos, pnl_oos, n_oos, "OOS")
        if m['n'] > 0:
            print(f"    {sp}={tv:>5.3f}: PF={m['pf']:.2f} WR={m['wr']:.0%} DD={m['mdd']:.1f}% N={m['n']}")

# ============================================================
# 6. SAVE EXPERIMENT RESULTS
# ============================================================
exp_results = []
for r in results:
    exp_results.append({
        'name': r['name'],
        'config': r['config'],
        'is_pf': r['is'].get('pf',0), 'is_wr': r['is'].get('wr',0),
        'is_mdd': r['is'].get('mdd',0), 'is_n': r['is'].get('n',0),
        'oos_pf': r['oos'].get('pf',0), 'oos_wr': r['oos'].get('wr',0),
        'oos_mdd': r['oos'].get('mdd',0), 'oos_n': r['oos'].get('n',0),
        'oos_ret': r['oos'].get('ret',0),
    })

with open('results.json', 'w') as f:
    json.dump(exp_results, f, indent=2, default=str)
print("\n\nSaved experiments/results.json")
print("\n✅ Experiments complete!")
