#!/usr/bin/env python3
"""
Optimized Strategy with realistic position sizing.
IS optimization + OOS validation. Fast vectorized backtest.
"""
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings, json
warnings.filterwarnings('ignore')

OOS_DATE = pd.Timestamp('2026-07-01')
INITIAL_EQUITY = 10000.0
SPREAD = 0.4  # $/oz
PER_CONTRACT = 100  # 1 lot = 100 oz
MAX_SIZE = 2.0  # Max 2 lots (realistic cap)

DATA_FILES = [f"XAUUSD_M5_{y}-{m:02d}.csv" for y in range(2024,2027) for m in range(1,13)
              if (y==2024 and m>=8) or (y==2025) or (y==2026 and m<=8)]

print("Loading & engineering...")
dfs = [pd.read_csv(f) for f in DATA_FILES]
df = pd.concat(dfs, ignore_index=True)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values('datetime').reset_index(drop=True)

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

EXCL = ['timestamp','date','time','open','high','low','close','volume','datetime',
        'lab_vol','lab_range','lab_candle','lab_trend','returns']
FEATS = [c for c in d.columns if c not in EXCL]
LABS = ['lab_vol','lab_range','lab_candle','lab_trend']
valid = d[FEATS+LABS].notna().all(axis=1)
dc = d[valid].reset_index(drop=True)
X_all = np.nan_to_num(dc[FEATS].values, nan=0.0)

is_m = dc['datetime'] < OOS_DATE
oos_m = dc['datetime'] >= OOS_DATE
X_is = X_all[is_m.values]
print(f"IS: {len(X_is)} | OOS: {len(X_all)-len(X_is)}")

# Train
print("Training models...")
xb = {'objective':'binary:logistic','eval_metric':'auc','max_depth':5,'learning_rate':0.05,
      'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.5,'reg_lambda':2.0,
      'min_child_weight':50,'n_estimators':300,'verbosity':0,'random_state':42}
xm = {**xb,'objective':'multi:softprob','eval_metric':'mlogloss','num_class':3}

mV = xgb.XGBClassifier(**xm).fit(X_is, dc.loc[is_m.values,'lab_vol'].astype(int).values, verbose=False)
mR = xgb.XGBClassifier(**xb).fit(X_is, dc.loc[is_m.values,'lab_range'].values, verbose=False)
mC = xgb.XGBClassifier(**xb).fit(X_is, dc.loc[is_m.values,'lab_candle'].values, verbose=False)
mT = xgb.XGBClassifier(**xb).fit(X_is, dc.loc[is_m.values,'lab_trend'].values, verbose=False)

vp = mV.predict_proba(X_all)
dc['vp_low']=vp[:,0]; dc['vp_norm']=vp[:,1]; dc['vp_high']=vp[:,2]
dc['vol_pred']=np.argmax(vp,1)
dc['range_p']=mR.predict_proba(X_all)[:,1]
dc['candle_p']=mC.predict_proba(X_all)[:,1]
dc['trend_p']=mT.predict_proba(X_all)[:,1]

print("OOS AUCs: Vol=%.3f Range=%.3f Candle=%.3f Trend=%.3f" % (
    roc_auc_score(dc.loc[oos_m.values,'lab_vol'].astype(int),vp[oos_m.values],multi_class='ovr',average='weighted'),
    roc_auc_score(dc.loc[oos_m.values,'lab_range'].values,dc.loc[oos_m.values,'range_p'].values),
    roc_auc_score(dc.loc[oos_m.values,'lab_candle'].values,dc.loc[oos_m.values,'candle_p'].values),
    roc_auc_score(dc.loc[oos_m.values,'lab_trend'].values,dc.loc[oos_m.values,'trend_p'].values)))

# ============================================================
# VECTORIZED BACKTEST ENGINE
# ============================================================
def backtest_vec(data, params):
    """Vectorized backtest - much faster."""
    n = len(data)
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    atr = data['atr_20'].values
    sma50 = data['sma_50'].values
    sma5 = data['sma_5'].values
    sma20 = data['sma_20'].values
    rsi14 = data['rsi_14'].values
    bb_pos = data['bb_pos_20'].values

    range_p = data['range_p'].values
    candle_p = data['candle_p'].values
    trend_p = data['trend_p'].values
    vol_pred = data['vol_pred'].values

    rt = params['rt']; ct = params['ct']; tt = params['tt']
    sl_mult = params['sl']; tp_mult = params['tp']
    risk = params['risk']; vsk = params.get('vsk', True)

    # Generate signals
    is_trend = trend_p >= tt
    long_cond = (
        (range_p >= rt) & (candle_p <= ct) &
        np.where(vsk, vol_pred != 0, True) &
        (
            (is_trend & (close > sma50) & (rsi14 > 40) & (rsi14 < 70) & (sma5 > sma20)) |
            (~is_trend & (bb_pos < 0.2) & (rsi14 < 35))
        )
    )
    short_cond = (
        (range_p >= rt) & (candle_p <= ct) &
        np.where(vsk, vol_pred != 0, True) &
        (
            (is_trend & (close < sma50) & (rsi14 > 30) & (rsi14 < 60) & (sma5 < sma20)) |
            (~is_trend & (bb_pos > 0.8) & (rsi14 > 65))
        )
    )

    # Simulate trades
    equity = INITIAL_EQUITY
    eq_curve = np.zeros(n)
    pnls = []
    position = 0  # 0=none, 1=long, -1=short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    size = 0.0

    for i in range(n):
        # Check exits
        if position != 0:
            if position == 1:  # Long
                if low[i] <= sl_price:
                    pnl = (sl_price - entry_price) * size * PER_CONTRACT - SPREAD * size * PER_CONTRACT
                    equity += pnl; pnls.append(pnl)
                    position = 0
                elif high[i] >= tp_price:
                    pnl = (tp_price - entry_price) * size * PER_CONTRACT - SPREAD * size * PER_CONTRACT
                    equity += pnl; pnls.append(pnl)
                    position = 0
            else:  # Short
                if high[i] >= sl_price:
                    pnl = (entry_price - sl_price) * size * PER_CONTRACT - SPREAD * size * PER_CONTRACT
                    equity += pnl; pnls.append(pnl)
                    position = 0
                elif low[i] <= tp_price:
                    pnl = (entry_price - tp_price) * size * PER_CONTRACT - SPREAD * size * PER_CONTRACT
                    equity += pnl; pnls.append(pnl)
                    position = 0

        # Check entries
        if position == 0:
            if long_cond[i]:
                position = 1
                entry_price = close[i]
                sl_price = entry_price - atr[i] * sl_mult
                tp_price = entry_price + atr[i] * tp_mult
                sl_dist = atr[i] * sl_mult
                size = min(max(equity * risk / (sl_dist * PER_CONTRACT), 0.01), MAX_SIZE)
            elif short_cond[i]:
                position = -1
                entry_price = close[i]
                sl_price = entry_price + atr[i] * sl_mult
                tp_price = entry_price - atr[i] * tp_mult
                sl_dist = atr[i] * sl_mult
                size = min(max(equity * risk / (sl_dist * PER_CONTRACT), 0.01), MAX_SIZE)

        eq_curve[i] = equity

    return eq_curve, pnls

def calc_metrics(eq, trades, label=""):
    if not trades: return f"{label}: No trades"
    pnls = np.array(trades)
    ret = (eq[-1] - INITIAL_EQUITY) / INITIAL_EQUITY * 100
    wr = np.mean(pnls > 0)
    gp = np.sum(pnls[pnls > 0])
    gl = abs(np.sum(pnls[pnls <= 0]))
    pf = gp / gl if gl > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    mdd = np.max(dd) * 100
    rets = np.diff(eq) / eq[:-1]
    rets = rets[rets != 0]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252 * 288) if len(rets) > 1 and np.std(rets) > 0 else 0
    return f"{label}: Ret={ret:+.1f}% Sharpe={sharpe:.2f} DD={mdd:.1f}% N={len(trades)} WR={wr:.0%} PF={pf:.2f} Eq=${eq[-1]:,.0f}"

# ============================================================
# OPTIMIZATION (focused grid, fast)
# ============================================================
is_data = dc[is_m.values].reset_index(drop=True)
oos_data = dc[oos_m.values].reset_index(drop=True)

print("\n" + "="*70)
print("PARAMETER OPTIMIZATION (IS)")
print("="*70)

# Focused grid based on initial test results
param_grid = {
    'rt': [0.55, 0.60, 0.65],
    'ct': [0.10, 0.15, 0.20],
    'tt': [0.50, 0.55],
    'sl': [1.5, 2.0, 2.5],
    'tp': [2.0, 2.5, 3.0],
    'risk': [0.01, 0.02],
    'vsk': [True],
}

from itertools import product
keys = list(param_grid.keys())
combos = list(product(*param_grid.values()))
print(f"Testing {len(combos)} combinations...")

best_score = -999
best_params = None
all_results = []

for combo in combos:
    p = dict(zip(keys, combo))
    try:
        eq, tr = backtest_vec(is_data, p)
        if len(tr) < 30: continue
        pnls = np.array(tr)
        ret = (eq[-1] - INITIAL_EQUITY) / INITIAL_EQUITY * 100
        wr = np.mean(pnls > 0)
        gp = np.sum(pnls[pnls > 0])
        gl = abs(np.sum(pnls[pnls <= 0]))
        pf = gp / gl if gl > 0 else 0
        peak = np.maximum.accumulate(eq)
        mdd = np.max((peak - eq) / peak) * 100
        rets = np.diff(eq) / eq[:-1]; rets = rets[rets != 0]
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252 * 288) if len(rets) > 1 else 0

        # Score: prefer Sharpe but penalize high drawdown
        score = sharpe - mdd * 2  # Penalize drawdown

        all_results.append({'params': p, 'ret': ret, 'sharpe': sharpe, 'mdd': mdd,
                           'n': len(tr), 'wr': wr, 'pf': pf, 'score': score})
        if score > best_score:
            best_score = score
            best_params = p
    except Exception as e:
        continue

all_results.sort(key=lambda x: x['score'], reverse=True)

print(f"\nTop 10 configurations (IS):")
print(f"{'#':<3} {'RT':>4} {'CT':>4} {'TT':>4} {'SL':>4} {'TP':>4} {'Rsk':>4} {'Ret%':>8} {'Shrp':>6} {'DD%':>6} {'N':>5} {'WR':>5} {'PF':>5}")
for i, r in enumerate(all_results[:10]):
    p = r['params']
    print(f"{i+1:<3} {p['rt']:>4.2f} {p['ct']:>4.2f} {p['tt']:>4.2f} {p['sl']:>4.1f} {p['tp']:>4.1f} {p['risk']:>4.2f} {r['ret']:>+7.1f}% {r['sharpe']:>6.2f} {r['mdd']:>5.1f}% {r['n']:>5} {r['wr']:>4.0%} {r['pf']:>5.2f}")

# Best params
bp = all_results[0]['params']
print(f"\nBest IS params: {json.dumps(bp)}")

# ============================================================
# OOS VALIDATION
# ============================================================
print("\n" + "="*70)
print("OOS VALIDATION (Best IS params)")
print("="*70)

oos_eq, oos_tr = backtest_vec(oos_data, bp)
print(calc_metrics(oos_eq, oos_tr, "OOS"))

# Also test top 3 configs on OOS
print("\nTop 3 configs on OOS:")
for i, r in enumerate(all_results[:3]):
    eq_oos, tr_oos = backtest_vec(oos_data, r['params'])
    print(calc_metrics(eq_oos, tr_oos, f"  #{i+1}"))

# ============================================================
# FULL PERIOD
# ============================================================
print("\n" + "="*70)
print("FULL PERIOD (IS + OOS)")
print("="*70)
full_data = dc.reset_index(drop=True)
full_eq, full_tr = backtest_vec(full_data, bp)
print(calc_metrics(full_eq, full_tr, "Full"))

# ============================================================
# MONTHLY OOS BREAKDOWN
# ============================================================
print("\n" + "="*70)
print("MONTHLY OOS BREAKDOWN")
print("="*70)
# Calculate monthly returns from equity curve
oos_dates = oos_data['datetime'].values
monthly = {}
for i in range(len(oos_eq)):
    mo = str(oos_dates[i])[:7]
    if mo not in monthly:
        monthly[mo] = {'start': oos_eq[i], 'end': oos_eq[i]}
    monthly[mo]['end'] = oos_eq[i]

prev_eq = INITIAL_EQUITY
for mo in sorted(monthly.keys()):
    m = monthly[mo]
    mret = (m['end'] - prev_eq) / prev_eq * 100
    print(f"  {mo}: {mret:+.2f}% (Eq: ${m['end']:,.0f})")
    prev_eq = m['end']

# ============================================================
# SAVE
# ============================================================
results = {
    'best_params': bp,
    'is_top10': [{k:v for k,v in r.items() if k!='params'} | {'params': r['params']} for r in all_results[:10]],
    'oos_metrics': {
        'ret': float((oos_eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100),
        'n_trades': len(oos_tr),
        'final_equity': float(oos_eq[-1]),
    },
    'full_metrics': {
        'ret': float((full_eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100),
        'n_trades': len(full_tr),
        'final_equity': float(full_eq[-1]),
    },
}
with open('strategy_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✅ Optimization complete! Results saved to strategy_results.json")
