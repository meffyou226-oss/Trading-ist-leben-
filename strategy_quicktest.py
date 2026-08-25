#!/usr/bin/env python3
"""
Quick Strategy Test: 3-4 strategy variants with fixed params.
Tests on IS + OOS. Fast execution, no optimization loop.
"""
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings, json
warnings.filterwarnings('ignore')

OOS_DATE = pd.Timestamp('2026-07-01')
INITIAL_EQUITY = 10000.0
SPREAD = 0.4
PER_CONTRACT = 100

DATA_FILES = [f"XAUUSD_M5_{y}-{m:02d}.csv" for y in range(2024,2027) for m in range(1,13)
              if (y==2024 and m>=8) or (y==2025) or (y==2026 and m<=8)]

print("Loading data...")
dfs = [pd.read_csv(f) for f in DATA_FILES]
df = pd.concat(dfs, ignore_index=True)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values('datetime').reset_index(drop=True)

print("Engineering features...")
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

print("OOS Model AUCs:")
print(f"  Vol:{roc_auc_score(dc.loc[oos_m.values,'lab_vol'].astype(int),vp[oos_m.values],multi_class='ovr',average='weighted'):.3f} "
      f"Range:{roc_auc_score(dc.loc[oos_m.values,'lab_range'].values,dc.loc[oos_m.values,'range_p'].values):.3f} "
      f"Candle:{roc_auc_score(dc.loc[oos_m.values,'lab_candle'].values,dc.loc[oos_m.values,'candle_p'].values):.3f} "
      f"Trend:{roc_auc_score(dc.loc[oos_m.values,'lab_trend'].values,dc.loc[oos_m.values,'trend_p'].values):.3f}")

# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest(data, params):
    eq = INITIAL_EQUITY; curve = []; trades = []; pos = None
    vsk = params.get('vsk', True)
    for i in range(len(data)):
        r = data.iloc[i]; atr = r['atr_20']; cl = r['close']
        # Exit
        if pos:
            ep,sl,tp,dr,sz = pos
            if dr==1:
                if r['low']<=sl:
                    pnl=(sl-ep)*sz*100-SPREAD*sz*100; eq+=pnl; trades.append(pnl); pos=None
                elif r['high']>=tp:
                    pnl=(tp-ep)*sz*100-SPREAD*sz*100; eq+=pnl; trades.append(pnl); pos=None
            else:
                if r['high']>=sl:
                    pnl=(ep-sl)*sz*100-SPREAD*sz*100; eq+=pnl; trades.append(pnl); pos=None
                elif r['low']<=tp:
                    pnl=(ep-tp)*sz*100-SPREAD*sz*100; eq+=pnl; trades.append(pnl); pos=None
        # Entry
        if pos is None and r['range_p']>=params['rt'] and r['candle_p']<=params['ct']:
            if vsk and r['vol_pred']==0: pass
            else:
                is_trend = r['trend_p']>=params['tt']
                sig = 0
                if is_trend:
                    if cl>r['sma_50'] and r['rsi_14']>40 and r['rsi_14']<70 and r['sma_5']>r['sma_20']: sig=1
                    if cl<r['sma_50'] and r['rsi_14']>30 and r['rsi_14']<60 and r['sma_5']<r['sma_20']: sig=-1
                else:
                    if r['bb_pos_20']<0.2 and r['rsi_14']<35: sig=1
                    if r['bb_pos_20']>0.8 and r['rsi_14']>65: sig=-1
                if sig!=0:
                    ep = cl
                    sld=atr*params['sl']; tpd=atr*params['tp']
                    if sig==1: sl=ep-sld; tp=ep+tpd
                    else: sl=ep+sld; tp=ep-tpd
                    sz=max(eq*params['risk']/(sld*100),0.01)
                    pos=(ep,sl,tp,sig,sz)
        curve.append(eq)
    return np.array(curve), trades

def calc_metrics(eq, trades, label):
    if not trades: return f"{label}: No trades"
    pnls=np.array(trades)
    ret=(eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    wr=np.mean(pnls>0)
    gp=np.sum(pnls[pnls>0]); gl=abs(np.sum(pnls[pnls<=0]))
    pf=gp/gl if gl>0 else 0
    peak=np.maximum.accumulate(eq); dd=(peak-eq)/peak; mdd=np.max(dd)*100
    rets=np.diff(eq)/eq[:-1]; rets=rets[rets!=0]
    sharpe=np.mean(rets)/np.std(rets)*np.sqrt(252*288) if len(rets)>1 else 0
    return f"{label}: Ret={ret:+.1f}% Sharpe={sharpe:.2f} DD={mdd:.1f}% N={len(trades)} WR={wr:.0%} PF={pf:.2f} Eq=${eq[-1]:,.0f}"

# ============================================================
# TEST STRATEGY VARIANTS
# ============================================================
is_data = dc[is_m.values].reset_index(drop=True)
oos_data = dc[oos_m.values].reset_index(drop=True)

strategies = {
    "A_TrendOnly":     {'rt':0.60,'ct':0.15,'tt':0.55,'sl':2.0,'tp':3.0,'risk':0.01,'vsk':True},
    "B_MeanRevOnly":   {'rt':0.60,'ct':0.15,'tt':0.40,'sl':1.5,'tp':2.5,'risk':0.01,'vsk':True},
    "C_Balanced":      {'rt':0.55,'ct':0.20,'tt':0.50,'sl':2.0,'tp':2.5,'risk':0.02,'vsk':True},
    "D_Aggressive":    {'rt':0.50,'ct':0.25,'tt':0.50,'sl':1.5,'tp':3.5,'risk':0.02,'vsk':False},
    "Cautious":     {'rt':0.65,'ct':0.10,'tt':0.55,'sl':2.5,'tp':2.0,'risk':0.01,'vsk':True},
}

print("\n" + "="*70)
print("STRATEGY TEST RESULTS")
print("="*70)
print(f"\nStrategy Params:")
for name, p in strategies.items():
    print(f"  {name}: RT={p['rt']} CT={p['ct']} TT={p['tt']} SL={p['sl']}x TP={p['tp']}x Risk={p['risk']}")

print(f"\n{'─'*70}")
print("IN-SAMPLE (Aug 2024 - Jun 2026)")
print(f"{'─'*70}")
for name, p in strategies.items():
    eq, tr = backtest(is_data, p)
    print(calc_metrics(eq, tr, f"  {name:<18}"))

print(f"\n{'─'*70}")
print("OUT-OF-SAMPLE (Jul-Aug 2026)")
print(f"{'─'*70}")
oos_results = {}
for name, p in strategies.items():
    eq, tr = backtest(oos_data, p)
    print(calc_metrics(eq, tr, f"  {name:<18}"))
    oos_results[name] = {'eq': eq, 'tr': tr, 'params': p}

# ============================================================
# SUMMARY TABLE
# ============================================================
print(f"\n{'='*70}")
print("COMPARISON: IS vs OOS")
print(f"{'='*70}")
print(f"{'Strategy':<18} {'IS Ret%':>8} {'OOS Ret%':>8} {'IS Sharpe':>10} {'OOS Sharpe':>10} {'IS N':>6} {'OOS N':>6}")
print(f"{'─'*70}")

comparison = []
for name, p in strategies.items():
    eq_is, tr_is = backtest(is_data, p)
    eq_oos, tr_oos = oos_data, None
    eq_oos, tr_oos = backtest(oos_data, p)
    ret_is = (eq_is[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    ret_oos = (eq_oos[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    rets_is = np.diff(eq_is)/eq_is[:-1]; rets_is=rets_is[rets_is!=0]
    sh_is = np.mean(rets_is)/np.std(rets_is)*np.sqrt(252*288) if len(rets_is)>1 else 0
    rets_oos = np.diff(eq_oos)/eq_oos[:-1]; rets_oos=rets_oos[rets_oos!=0]
    sh_oos = np.mean(rets_oos)/np.std(rets_oos)*np.sqrt(252*288) if len(rets_oos)>1 else 0
    print(f"{name:<18} {ret_is:>+7.1f}% {ret_oos:>+7.1f}% {sh_is:>10.2f} {sh_oos:>10.2f} {len(tr_is):>6} {len(tr_oos):>6}")
    comparison.append({'name':name,'params':p,'is_ret':ret_is,'oos_ret':ret_oos,'is_sharpe':sh_is,'oos_sharpe':sh_oos,'is_n':len(tr_is),'oos_n':len(tr_oos)})

with open('strategy_comparison.json','w') as f:
    json.dump(comparison, f, indent=2, default=str)

print(f"\n✅ Quick strategy test complete! Results saved to strategy_comparison.json")
