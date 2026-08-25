#!/usr/bin/env python3
"""
Regime-Adaptive Trading Strategy using 4 XGBoost Models.
IS Optimization + OOS Validation.
"""
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from itertools import product
import warnings, json
warnings.filterwarnings('ignore')

OOS_DATE = pd.Timestamp('2026-07-01')
INITIAL_EQUITY = 10000.0
SPREAD = 0.4  # $/oz spread
PER_CONTRACT = 100

DATA_FILES = [f"XAUUSD_M5_{y}-{m:02d}.csv" for y in range(2024,2027) for m in range(1,13)
              if (y==2024 and m>=8) or (y==2025) or (y==2026 and m<=8)]

print("Loading data...")
dfs = [pd.read_csv(f) for f in DATA_FILES]
df = pd.concat(dfs, ignore_index=True)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values('datetime').reset_index(drop=True)

# --- Feature Engineering ---
print("Engineering features...")
d = df.copy()
d['returns'] = d['close'].pct_change()
d['range'] = d['high'] - d['low']
d['body'] = abs(d['close'] - d['open'])
d['bb_body_ratio'] = d['body'] / (d['range'] + 1e-10)
for p in [5,10,20,50]: d[f'atr_{p}'] = d['range'].rolling(p).mean()
d['atr_ratio'] = d['atr_5']/(d['atr_20']+1e-10)
for p in [5,10,20,50,100,200]: d[f'sma_{p}'] = d['close'].rolling(p).mean()
for p in [5,10,20,50,100,200]:
    atr_ref = d[f'atr_{p}'] if f'atr_{p}' in d.columns else d['atr_50']
    d[f'dist_sma_{p}'] = (d['close']-d[f'sma_{p}'])/(atr_ref+1e-10)
for p in [10,20]:
    up = d['high'].diff(); dn = -d['low'].diff()
    up[up<0]=0; dn[dn<0]=0
    atr_s = d['range'].rolling(p).sum()
    pdi = up.rolling(p).sum()/(atr_s+1e-10)*100
    mdi = dn.rolling(p).sum()/(atr_s+1e-10)*100
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
    d[f'bb_upper_{p}'] = s+2*sdev; d[f'bb_lower_{p}'] = s-2*sdev
    d[f'bb_pos_{p}'] = (d['close']-d[f'bb_lower_{p}'])/(4*sdev+1e-10)
for h in [1,3,5,10,15,20]: d[f'ret_{h}bar'] = d['close'].pct_change(h)
d['vol_ratio'] = d['volume']/(d['volume'].rolling(20).mean()+1e-10)
d['hour'] = d['datetime'].dt.hour
d['is_overlap'] = ((d['hour']>=12)&(d['hour']<=16)).astype(int)
d['drawdown'] = (d['close']-d['high'].rolling(50).max())/(d['high'].rolling(50).max()+1e-10)

# --- Labels ---
future_atr = d['range'].rolling(10).mean().shift(-10)
p33, p66 = future_atr.quantile(0.33), future_atr.quantile(0.66)
d['lab_vol'] = pd.cut(future_atr, bins=[-np.inf,p33,p66,np.inf], labels=[0,1,2])
fr = d['close'].rolling(10).apply(lambda x: x.max()-x.min(), raw=True).shift(-10)/(d['atr_20']+1e-10)
d['lab_range'] = (fr > fr.median()).astype(int)
d['lab_candle'] = (d['body'].rolling(5).max().shift(-5) > 2*d['atr_20']).astype(int)
d['er'] = d['close'].rolling(20).apply(lambda x: abs(x.iloc[-1]-x.iloc[0])/(x.diff().abs().sum()+1e-10), raw=False).shift(-10)
d['lab_trend'] = ((d['er']>d['er'].median())&(d['adx_20'].shift(-10)>25)).astype(int)

# --- Prepare ---
EXCL = ['timestamp','date','time','open','high','low','close','volume','datetime',
        'lab_vol','lab_range','lab_candle','lab_trend','returns']
FEATS = [c for c in d.columns if c not in EXCL]
LABS = ['lab_vol','lab_range','lab_candle','lab_trend']
valid = d[FEATS+LABS].notna().all(axis=1)
dc = d[valid].reset_index(drop=True)
X_all = np.nan_to_num(dc[FEATS].values, nan=0.0)

is_m = dc['datetime'] < OOS_DATE
oos_m = dc['datetime'] >= OOS_DATE
X_is, X_oos = X_all[is_m.values], X_all[oos_m.values]
print(f"IS: {len(X_is)} | OOS: {len(X_oos)}")

# --- Train Models on IS ---
print("Training models on IS data...")
xb = {'objective':'binary:logistic','eval_metric':'auc','max_depth':5,'learning_rate':0.05,
      'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.5,'reg_lambda':2.0,
      'min_child_weight':50,'n_estimators':500,'verbosity':0,'random_state':42}
xm = {**xb,'objective':'multi:softprob','eval_metric':'mlogloss','num_class':3}

yv = dc.loc[is_m.values,'lab_vol'].values.astype(int)
yr = dc.loc[is_m.values,'lab_range'].values.astype(int)
yc = dc.loc[is_m.values,'lab_candle'].values.astype(int)
yt = dc.loc[is_m.values,'lab_trend'].values.astype(int)

mV = xgb.XGBClassifier(**xm).fit(X_is, yv, verbose=False)
mR = xgb.XGBClassifier(**xb).fit(X_is, yr, verbose=False)
mC = xgb.XGBClassifier(**xb).fit(X_is, yc, verbose=False)
mT = xgb.XGBClassifier(**xb).fit(X_is, yt, verbose=False)

# --- Predictions ---
vp = mV.predict_proba(X_all)
dc['vp_low']=vp[:,0]; dc['vp_norm']=vp[:,1]; dc['vp_high']=vp[:,2]; dc['vol_pred']=np.argmax(vp,1)
dc['range_p']=mR.predict_proba(X_all)[:,1]
dc['candle_p']=mC.predict_proba(X_all)[:,1]
dc['trend_p']=mT.predict_proba(X_all)[:,1]

print("\nOOS Model AUCs:")
print(f"  Vol: {roc_auc_score(dc.loc[oos_m.values,'lab_vol'].astype(int), vp[oos_m.values], multi_class='ovr', average='weighted'):.4f}")
print(f"  Range: {roc_auc_score(dc.loc[oos_m.values,'lab_range'].values, dc.loc[oos_m.values,'range_p'].values):.4f}")
print(f"  Candle: {roc_auc_score(dc.loc[oos_m.values,'lab_candle'].values, dc.loc[oos_m.values,'candle_p'].values):.4f}")
print(f"  Trend: {roc_auc_score(dc.loc[oos_m.values,'lab_trend'].values, dc.loc[oos_m.values,'trend_p'].values):.4f}")

# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest(data, p):
    equity = INITIAL_EQUITY
    eq_curve = []; trades = []; pos = None
    for i in range(len(data)):
        r = data.iloc[i]; atr = r['atr_20']; close = r['close']
        # Exit
        if pos is not None:
            ep, sl, tp, dr, sz = pos['ep'],pos['sl'],pos['tp'],pos['d'],pos['sz']
            if dr==1:
                if r['low']<=sl:
                    pnl=(sl-ep)*sz*PER_CONTRACT-SPREAD*sz*PER_CONTRACT; equity+=pnl
                    trades.append({'pnl':pnl,'res':'SL'}); pos=None
                elif r['high']>=tp:
                    pnl=(tp-ep)*sz*PER_CONTRACT-SPREAD*sz*PER_CONTRACT; equity+=pnl
                    trades.append({'pnl':pnl,'res':'TP'}); pos=None
            else:
                if r['high']>=sl:
                    pnl=(ep-sl)*sz*PER_CONTRACT-SPREAD*sz*PER_CONTRACT; equity+=pnl
                    trades.append({'pnl':pnl,'res':'SL'}); pos=None
                elif r['low']<=tp:
                    pnl=(ep-tp)*sz*PER_CONTRACT-SPREAD*sz*PER_CONTRACT; equity+=pnl
                    trades.append({'pnl':pnl,'res':'TP'}); pos=None
        # Entry
        if pos is None:
            if (r['range_p']>=p['rt'] and r['candle_p']<=p['ct'] and
                not (p['vsk'] and r['vol_pred']==0)):
                is_trend = r['trend_p']>=p['tt']
                long_s = short_s = False
                if is_trend:
                    if close>r['sma_50'] and 40<r['rsi_14']<70 and r['sma_5']>r['sma_20']: long_s=True
                    if close<r['sma_50'] and 30<r['rsi_14']<60 and r['sma_5']<r['sma_20']: short_s=True
                else:
                    if r['bb_pos_20']<0.2 and r['rsi_14']<35: long_s=True
                    if r['bb_pos_20']>0.8 and r['rsi_14']>65: short_s=True
                if long_s or short_s:
                    dr = 1 if long_s else -1; ep = close
                    sld = atr*p['sl']; tpd = atr*p['tp']
                    sl = ep-sld if dr==1 else ep+sld
                    tp = ep+tpd if dr==1 else ep-tpd
                    risk = equity*p['risk']
                    sz = max(min(risk/(sld*PER_CONTRACT), equity/(ep*PER_CONTRACT)), 0.01)
                    pos={'ep':ep,'sl':sl,'tp':tp,'d':dr,'sz':sz}
        eq_curve.append(equity)
    return np.array(eq_curve), trades

def metrics(eq, trades):
    if not trades: return {'sharpe':0,'mdd':0,'ret':0,'n':0,'wr':0,'pf':0}
    pnls=[t['pnl'] for t in trades]
    w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
    ret=(eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    n=len(trades); wr=len(w)/n if n else 0
    gp=sum(w) if w else 0; gl=abs(sum(l)) if l else 1
    pf=gp/gl if gl>0 else 0
    peak=np.maximum.accumulate(eq); dd=(peak-eq)/peak
    mdd=np.max(dd)*100
    rets=np.diff(eq)/eq[:-1]; rets=rets[rets!=0]
    sharpe=np.mean(rets)/np.std(rets)*np.sqrt(252*288) if len(rets)>1 and np.std(rets)>0 else 0
    return {'sharpe':sharpe,'mdd':mdd,'ret':ret,'n':n,'wr':wr,'pf':pf}

# ============================================================
# OPTIMIZATION
# ============================================================
print("\n" + "="*70)
print("PARAMETER OPTIMIZATION")
print("="*70)

is_data = dc[is_m.values].reset_index(drop=True)
param_grid = {
    'rt': [0.55, 0.60, 0.65, 0.70],
    'ct': [0.10, 0.15, 0.20],
    'tt': [0.50, 0.55, 0.60],
    'sl': [1.5, 2.0, 2.5],
    'tp': [2.0, 2.5, 3.0, 3.5],
    'risk': [0.01, 0.02],
    'vsk': [True],
}
keys = list(param_grid.keys())
combos = list(product(*param_grid.values()))
np.random.seed(42)
if len(combos)>300:
    combos = [combos[i] for i in np.random.choice(len(combos), 300, replace=False)]
print(f"Testing {len(combos)} combinations...")

best_sharpe = -999; best_p = None; all_res = []
for combo in combos:
    p = dict(zip(keys, combo))
    try:
        eq, tr = backtest(is_data, p)
        m = metrics(eq, tr)
        if m['n'] < 20: continue  # Need minimum trades
        m['params'] = p
        all_res.append(m)
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']; best_p = p
    except: continue

print(f"\nBest IS Parameters: {json.dumps(best_p, indent=2)}")
is_metrics = metrics(*backtest(is_data, best_p))
print(f"IS Metrics: Sharpe={is_metrics['sharpe']:.3f}, Return={is_metrics['ret']:.1f}%, "
      f"MaxDD={is_metrics['mdd']:.1f}%, Trades={is_metrics['n']}, "
      f"WinRate={is_metrics['wr']:.1%}, PF={is_metrics['pf']:.2f}")

# Top 5 by Sharpe
all_res.sort(key=lambda x: x['sharpe'], reverse=True)
print("\nTop 5 configurations (IS):")
for i, r in enumerate(all_res[:5]):
    print(f"  {i+1}. Sharpe={r['sharpe']:.3f} Ret={r['ret']:.1f}% DD={r['mdd']:.1f}% "
          f"N={r['n']} WR={r['wr']:.0%} PF={r['pf']:.2f} | {r['params']}")

# ============================================================
# OOS VALIDATION
# ============================================================
print("\n" + "="*70)
print("OOS VALIDATION (Best IS Parameters)")
print("="*70)

oos_data = dc[oos_m.values].reset_index(drop=True)
oos_eq, oos_trades = backtest(oos_data, best_p)
oos_metrics = metrics(oos_eq, oos_trades)

print(f"\nOOS Results (Jul-Aug 2026):")
print(f"  Total Return: {oos_metrics['ret']:.2f}%")
print(f"  Sharpe Ratio: {oos_metrics['sharpe']:.3f}")
print(f"  Max Drawdown: {oos_metrics['mdd']:.2f}%")
print(f"  Total Trades: {oos_metrics['n']}")
print(f"  Win Rate: {oos_metrics['wr']:.1%}")
print(f"  Profit Factor: {oos_metrics['pf']:.2f}")
print(f"  Final Equity: ${oos_eq[-1]:,.2f}")

# Trade breakdown
if oos_trades:
    longs = [t['pnl'] for t in oos_trades if t.get('d','')!='']
    tps = [t for t in oos_trades if t['res']=='TP']
    sls = [t for t in oos_trades if t['res']=='SL']
    print(f"\n  TP hits: {len(tps)} | SL hits: {len(sls)}")
    print(f"  Avg TP: ${np.mean([t['pnl'] for t in tps]):.2f}" if tps else "")
    print(f"  Avg SL: ${np.mean([t['pnl'] for t in sls]):.2f}" if sls else "")

# ============================================================
# FULL PERIOD (IS+OOS with best params)
# ============================================================
print("\n" + "="*70)
print("FULL PERIOD RESULTS (IS+OOS combined)")
print("="*70)
full_data = dc.reset_index(drop=True)
full_eq, full_trades = backtest(full_data, best_p)
full_metrics = metrics(full_eq, full_trades)
print(f"  Total Return: {full_metrics['ret']:.2f}%")
print(f"  Sharpe Ratio: {full_metrics['sharpe']:.3f}")
print(f"  Max Drawdown: {full_metrics['mdd']:.2f}%")
print(f"  Total Trades: {full_metrics['n']}")
print(f"  Win Rate: {full_metrics['wr']:.1%}")
print(f"  Profit Factor: {full_metrics['pf']:.2f}")

# ============================================================
# MONTHLY BREAKDOWN
# ============================================================
print("\n" + "="*70)
print("MONTHLY OOS BREAKDOWN")
print("="*70)
if oos_trades:
    oos_data_copy = oos_data.copy()
    oos_data_copy['equity'] = oos_eq
    monthly = {}
    for t in oos_trades:
        if 'exit' in t:
            exit_dt = t.get('exit')
            if exit_dt is not None:
                mo = pd.Timestamp(exit_dt).strftime('%Y-%m')
                if mo not in monthly: monthly[mo] = []
                monthly[mo].append(t['pnl'])
    # Use equity curve for monthly returns instead
    oos_data_copy['date'] = oos_data_copy['datetime'].dt.to_period('M')
    monthly_eq = oos_data_copy.groupby('date')['equity'].last()
    prev = INITIAL_EQUITY
    for date, eq_val in monthly_eq.items():
        mret = (eq_val - prev) / prev * 100
        n_tr = len([t for t in oos_trades if 'exit' in t and t['exit'] is not None and pd.Timestamp(t['exit']).strftime('%Y-%m') == str(date)])
        print(f"  {date}: Return={mret:+.2f}%, Trades={n_tr}")
        prev = eq_val

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'best_params': best_p,
    'is_metrics': {k:v for k,v in is_metrics.items() if k!='params'},
    'oos_metrics': oos_metrics,
    'full_metrics': full_metrics,
}
with open('strategy_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to strategy_results.json")
print("\n✅ Strategy backtest complete!")
