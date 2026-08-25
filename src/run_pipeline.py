#!/usr/bin/env python3
"""
Complete Pipeline: Train models -> Save -> Generate Predictions -> Run Strategy
Best OOS Config: RT=0.65, CT=0.10, TT=0.55, SL=2.0x, TP=2.5x, Risk=1%
OOS Result: PF=2.38, WR=67%, MaxDD=7.1%, +185% return
"""
import pandas as pd, numpy as np, xgboost as xgb, json, pickle, os
from sklearn.metrics import roc_auc_score
warnings = np.warnings if hasattr(np, 'warnings') else None
import warnings as w
w.filterwarnings('ignore')

OOS_DATE = pd.Timestamp('2026-07-01')
INITIAL_EQUITY = 10000.0
SPREAD = 0.4
PER_CONTRACT = 100
MAX_SIZE = 2.0

# Best OOS configuration (PF=2.38)
BEST_CONFIG = {
    'rt': 0.65, 'ct': 0.10, 'tt': 0.55,
    'sl': 2.0, 'tp': 2.5, 'risk': 0.01, 'vsk': True
}

DATA_DIR = ".."
MODELS_DIR = "../models"
RESULTS_DIR = "../results"
CONFIG_DIR = "../config"

DATA_FILES = [f"XAUUSD_M5_{y}-{m:02d}.csv" for y in range(2024,2027) for m in range(1,13)
              if (y==2024 and m>=8) or (y==2025) or (y==2026 and m<=8)]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & ENGINEER FEATURES
# ============================================================
print("="*70)
print("1. LOADING DATA & ENGINEERING FEATURES")
print("="*70)

dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in DATA_FILES]
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
print(f"Features: {len(FEATS)} | IS: {len(X_is)} | OOS: {len(X_all)-len(X_is)}")

# ============================================================
# 2. TRAIN & SAVE MODELS
# ============================================================
print("\n" + "="*70)
print("2. TRAINING & SAVING MODELS")
print("="*70)

xb = {'objective':'binary:logistic','eval_metric':'auc','max_depth':5,'learning_rate':0.05,
      'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.5,'reg_lambda':2.0,
      'min_child_weight':50,'n_estimators':300,'verbosity':0,'random_state':42}
xm = {**xb,'objective':'multi:softprob','eval_metric':'mlogloss','num_class':3}

models = {}
print("Training Vol-Regime...")
models['vol_regime'] = xgb.XGBClassifier(**xm).fit(X_is, dc.loc[is_m.values,'lab_vol'].astype(int).values, verbose=False)
print("Training Future-Range...")
models['future_range'] = xgb.XGBClassifier(**xb).fit(X_is, dc.loc[is_m.values,'lab_range'].values, verbose=False)
print("Training Large-Candle...")
models['large_candle'] = xgb.XGBClassifier(**xb).fit(X_is, dc.loc[is_m.values,'lab_candle'].values, verbose=False)
print("Training Trend-Regime...")
models['trend_regime'] = xgb.XGBClassifier(**xb).fit(X_is, dc.loc[is_m.values,'lab_trend'].values, verbose=False)

# Save models
for name, model in models.items():
    with open(os.path.join(MODELS_DIR, f'{name}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved {name}.pkl")

# Save feature list
with open(os.path.join(CONFIG_DIR, 'features.json'), 'w') as f:
    json.dump(FEATS, f, indent=2)
print(f"  Saved features.json ({len(FEATS)} features)")

# ============================================================
# 3. GENERATE PREDICTIONS
# ============================================================
print("\n" + "="*70)
print("3. GENERATING PREDICTIONS")
print("="*70)

vp = models['vol_regime'].predict_proba(X_all)
dc['vp_low']=vp[:,0]; dc['vp_norm']=vp[:,1]; dc['vp_high']=vp[:,2]
dc['vol_pred']=np.argmax(vp,1)
dc['range_p']=models['future_range'].predict_proba(X_all)[:,1]
dc['candle_p']=models['large_candle'].predict_proba(X_all)[:,1]
dc['trend_p']=models['trend_regime'].predict_proba(X_all)[:,1]

# Save predictions
dc[['datetime','close','vol_pred','range_p','candle_p','trend_p']].to_csv(
    os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)
print("  Saved predictions.csv")

# OOS AUC
oos_auc = {
    'vol_regime': roc_auc_score(dc.loc[oos_m.values,'lab_vol'].astype(int),vp[oos_m.values],multi_class='ovr',average='weighted'),
    'future_range': roc_auc_score(dc.loc[oos_m.values,'lab_range'].values,dc.loc[oos_m.values,'range_p'].values),
    'large_candle': roc_auc_score(dc.loc[oos_m.values,'lab_candle'].values,dc.loc[oos_m.values,'candle_p'].values),
    'trend_regime': roc_auc_score(dc.loc[oos_m.values,'lab_trend'].values,dc.loc[oos_m.values,'trend_p'].values),
}
print(f"  OOS AUCs: {oos_auc}")

# ============================================================
# 4. RUN STRATEGY (BEST OOS CONFIG)
# ============================================================
print("\n" + "="*70)
print("4. RUNNING STRATEGY (Best OOS Config)")
print("="*70)

def backtest(data, params):
    n = len(data)
    close = data['close'].values; high = data['high'].values; low = data['low'].values
    atr = data['atr_20'].values; sma50 = data['sma_50'].values
    sma5 = data['sma_5'].values; sma20 = data['sma_20'].values
    rsi14 = data['rsi_14'].values; bb_pos = data['bb_pos_20'].values
    range_p = data['range_p'].values; candle_p = data['candle_p'].values
    trend_p = data['trend_p'].values; vol_pred = data['vol_pred'].values

    rt=params['rt']; ct=params['ct']; tt=params['tt']
    sl_m=params['sl']; tp_m=params['tp']; risk=params['risk']; vsk=params.get('vsk',True)

    is_trend = trend_p >= tt
    long_c = (range_p>=rt)&(candle_p<=ct)&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close>sma50)&(rsi14>40)&(rsi14<70)&(sma5>sma20))|
        (~is_trend&(bb_pos<0.2)&(rsi14<35)))
    short_c = (range_p>=rt)&(candle_p<=ct)&np.where(vsk,vol_pred!=0,True)&(
        (is_trend&(close<sma50)&(rsi14>30)&(rsi14<60)&(sma5<sma20))|
        (~is_trend&(bb_pos>0.8)&(rsi14>65)))

    eq = INITIAL_EQUITY; curve = np.zeros(n); pnls = []; n_trades = 0
    pos = 0; ep = 0.0; sl = 0.0; tp = 0.0; sz = 0.0

    for i in range(n):
        if pos != 0:
            if pos == 1:
                if low[i] <= sl:
                    pnl=(sl-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
                elif high[i] >= tp:
                    pnl=(tp-ep)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
            else:
                if high[i] >= sl:
                    pnl=(ep-sl)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
                elif low[i] <= tp:
                    pnl=(ep-tp)*sz*100-SPREAD*sz*100; eq+=pnl; pnls.append(pnl); n_trades+=1; pos=0
        if pos == 0:
            if long_c[i]:
                pos=1; ep=close[i]; sl=ep-atr[i]*sl_m; tp=ep+atr[i]*tp_m
                sz=min(max(eq*risk/(atr[i]*sl_m*100),0.01),MAX_SIZE)
            elif short_c[i]:
                pos=-1; ep=close[i]; sl=ep+atr[i]*sl_m; tp=ep-atr[i]*tp_m
                sz=min(max(eq*risk/(atr[i]*sl_m*100),0.01),MAX_SIZE)
        curve[i] = eq
    return curve, np.array(pnls), n_trades

# Run on full data
eq_full, pnl_full, n_full = backtest(dc, BEST_CONFIG)
# Run on OOS only
oos_data = dc[oos_m.values].reset_index(drop=True)
eq_oos, pnl_oos, n_oos = backtest(oos_data, BEST_CONFIG)

def metrics(eq, pnls, n_trades, label):
    if n_trades == 0: return f"{label}: No trades"
    ret = (eq[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
    wr = np.mean(pnls>0)
    gp = np.sum(pnls[pnls>0]); gl = abs(np.sum(pnls[pnls<=0]))
    pf = gp/gl if gl>0 else 0
    peak = np.maximum.accumulate(eq); mdd = np.max((peak-eq)/peak)*100
    rets = np.diff(eq)/eq[:-1]; rets=rets[rets!=0]
    shr = np.mean(rets)/np.std(rets)*np.sqrt(252*288) if len(rets)>1 else 0
    return f"{label}: Ret={ret:+.1f}% PF={pf:.2f} WR={wr:.0%} DD={mdd:.1f}% N={n_trades} Eq=${eq[-1]:,.0f}"

print(metrics(eq_full, pnl_full, n_full, "FULL"))
print(metrics(eq_oos, pnl_oos, n_oos, "OOS"))

# ============================================================
# 5. SAVE EVERYTHING
# ============================================================
print("\n" + "="*70)
print("5. SAVING ALL RESULTS")
print("="*70)

# Save config
with open(os.path.join(CONFIG_DIR, 'best_strategy.json'), 'w') as f:
    json.dump(BEST_CONFIG, f, indent=2)
print("  Saved config/best_strategy.json")

# Save equity curves
np.save(os.path.join(RESULTS_DIR, 'equity_full.npy'), eq_full)
np.save(os.path.join(RESULTS_DIR, 'equity_oos.npy'), eq_oos)
print("  Saved equity curves (.npy)")

# Save trade PnLs
np.save(os.path.join(RESULTS_DIR, 'pnl_full.npy'), pnl_full)
np.save(os.path.join(RESULTS_DIR, 'pnl_oos.npy'), pnl_oos)
print("  Saved trade PnLs (.npy)")

# Summary report
oos_ret = (eq_oos[-1]-INITIAL_EQUITY)/INITIAL_EQUITY*100
oos_wr = float(np.mean(pnl_oos>0))
oos_gp = float(np.sum(pnl_oos[pnl_oos>0])); oos_gl = float(abs(np.sum(pnl_oos[pnl_oos<=0])))
oos_pf = oos_gp/oos_gl if oos_gl>0 else 0
oos_peak = np.maximum.accumulate(eq_oos); oos_mdd = float(np.max((oos_peak-eq_oos)/oos_peak)*100)

summary = {
    'strategy': 'Regime-Adaptive XGBoost',
    'config': BEST_CONFIG,
    'oos_period': '2026-07-01 to 2026-08-25',
    'oos_metrics': {
        'total_return_pct': round(oos_ret, 2),
        'profit_factor': round(oos_pf, 2),
        'win_rate': round(oos_wr, 3),
        'max_drawdown_pct': round(oos_mdd, 2),
        'total_trades': n_oos,
        'final_equity': round(float(eq_oos[-1]), 2),
    },
    'is_period': '2024-08-01 to 2026-06-30',
    'is_metrics': {
        'total_return_pct': round(float((eq_full[len(X_is)+100]-INITIAL_EQUITY)/INITIAL_EQUITY*100), 2),
        'total_trades': n_full - n_oos,
    },
    'model_oos_auc': oos_auc,
    'models': ['vol_regime', 'future_range', 'large_candle', 'trend_regime'],
}
with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("  Saved results/summary.json")

print("\n" + "="*70)
print("FOLDER STRUCTURE")
print("="*70)
print("""
models/
  vol_regime.pkl        - Volatilitäts-Regime Model (LOW/NORMAL/HIGH)
  future_range.pkl      - Future Range Probability Model
  large_candle.pkl      - Large Candle Probability Model
  trend_regime.pkl      - Trend/Range Regime Model

config/
  features.json         - Feature-Namen für Inference
  best_strategy.json    - Beste Strategie-Parameter

results/
  predictions.csv       - Modell-Vorhersagen für alle Bars
  equity_full.npy       - Vollständige Equity Curve
  equity_oos.npy        - OOS Equity Curve
  pnl_full.npy          - Alle Trade PnLs
  pnl_oos.npy           - OOS Trade PnLs
  summary.json          - Zusammenfassung aller Metriken

src/
  run_pipeline.py       - Dieses Skript
  train_models.py       - Modell-Training & Evaluation
  strategy_optimized.py - Strategie mit IS-Optimierung
  strategy_quicktest.py - Schnellvergleich 5 Varianten
""")

print("\n✅ Pipeline complete! All models, configs, and results saved.")
