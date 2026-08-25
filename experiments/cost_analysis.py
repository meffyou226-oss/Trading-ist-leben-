#!/usr/bin/env python3
"""
Realistic Cost Analysis: Test PF sensitivity to spreads, commissions, slippage.
Also tests longer OOS periods and walk-forward to check overfitting.
"""
import pandas as pd, numpy as np, xgboost as xgb, pickle, json, os
import warnings
warnings.filterwarnings('ignore')

OOS_DATE = pd.Timestamp('2026-07-01')
INITIAL_EQUITY = 10000.0
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
# REALISTIC BACKTEST WITH COSTS
# ============================================================
def backtest_costs(data, config, spread_per_oz=0.4, commission_per_lot=0.0, slippage_per_oz=0.0):
    """Backtest with realistic transaction costs."""
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

    # Total cost per trade (round trip)
    # Spread: paid once (entry), Commission: round trip, Slippage: entry + exit
    cost_per_trade = spread_per_oz + slippage_per_oz  # per oz, entry
    cost_per_trade_exit = slippage_per_oz  # per oz, exit
    commission = commission_per_lot  # per lot (100 oz)

    eq = INITIAL_EQUITY; curve = np.zeros(n); pnls = []; n_trades = 0
    pos = 0; ep = 0.0; sl_val = 0.0; tp_val = 0.0; sz = 0.0

    for i in range(n):
        if pos != 0:
            if pos == 1:
                if low[i] <= sl_val:
                    pnl = (sl_val - ep) * sz * PER_CONTRACT - cost_per_trade * sz * PER_CONTRACT - cost_per_trade_exit * sz * PER_CONTRACT - commission * sz
                    eq += pnl; pnls.append(pnl); n_trades += 1; pos = 0
                elif high[i] >= tp_val:
                    pnl = (tp_val - ep) * sz * PER_CONTRACT - cost_per_trade * sz * PER_CONTRACT - cost_per_trade_exit * sz * PER_CONTRACT - commission * sz
                    eq += pnl; pnls.append(pnl); n_trades += 1; pos = 0
            else:
                if high[i] >= sl_val:
                    pnl = (ep - sl_val) * sz * PER_CONTRACT - cost_per_trade * sz * PER_CONTRACT - cost_per_trade_exit * sz * PER_CONTRACT - commission * sz
                    eq += pnl; pnls.append(pnl); n_trades += 1; pos = 0
                elif low[i] <= tp_val:
                    pnl = (ep - tp_val) * sz * PER_CONTRACT - cost_per_trade * sz * PER_CONTRACT - cost_per_trade_exit * sz * PER_CONTRACT - commission * sz
                    eq += pnl; pnls.append(pnl); n_trades += 1; pos = 0

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
                sz = min(max(eq*risk/(sl_dist*PER_CONTRACT), 0.01), MAX_SIZE)
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
# COST SENSITIVITY ANALYSIS
# ============================================================
oos_data = dc[oos_m.values].reset_index(drop=True)

# Best configs to test
configs = {
    'ultra_pf': {'rt':0.72,'ct':0.08,'sl':1.5,'tp':3.0,'tt':0.50,'risk':0.02,'vsk':True,'session_filter':'no_asian'},
    'high_pf_freq': {'rt':0.65,'ct':0.08,'sl':1.0,'tp':2.5,'tt':0.50,'risk':0.02,'vsk':True,'session_filter':'no_asian'},
    'balanced': {'rt':0.65,'ct':0.10,'sl':1.0,'tp':2.5,'tt':0.50,'risk':0.01,'vsk':True,'session_filter':'no_asian'},
}

# Cost scenarios (per oz for spread/slippage, per lot for commission)
# User: Commission = $0.28 per 0.05 lots = $5.60 per standard lot
cost_scenarios = {
    'ideal':        {'spread': 0.3, 'commission': 0.0, 'slippage': 0.0},
    'realistic_low': {'spread': 0.4, 'commission': 2.8, 'slippage': 0.05},
    'realistic':    {'spread': 0.5, 'commission': 5.6, 'slippage': 0.1},
    'realistic_high': {'spread': 0.8, 'commission': 5.6, 'slippage': 0.2},
    'expensive':    {'spread': 1.0, 'commission': 7.0, 'slippage': 0.3},
    'worst_case':   {'spread': 1.5, 'commission': 10.0, 'slippage': 0.5},
}

print("\n" + "="*70)
print("COST SENSITIVITY ANALYSIS")
print("="*70)

for cfg_name, cfg in configs.items():
    print(f"\n{'─'*70}")
    print(f"Config: {cfg_name}")
    print(f"{'─'*70}")
    print(f"{'Scenario':<20} {'Spread':>6} {'Comm':>6} {'Slip':>6} | {'PF':>6} {'WR':>5} {'DD%':>5} {'N':>5} {'Ret%':>8}")
    print(f"{'─'*70}")

    for scen_name, costs in cost_scenarios.items():
        eq, pnls, n = backtest_costs(oos_data, cfg,
                                      spread_per_oz=costs['spread'],
                                      commission_per_lot=costs['commission'],
                                      slippage_per_oz=costs['slippage'])
        m = metrics(eq, pnls, n)
        if m:
            print(f"{scen_name:<20} {costs['spread']:>5.1f}$ {costs['commission']:>5.1f}$ {costs['slippage']:>5.2f}$ | "
                  f"{m['pf']:>6.2f} {m['wr']:>4.0%} {m['mdd']:>4.1f} {m['n']:>5} {m['ret']:>+7.1f}%")

# ============================================================
# LONGER OOS PERIOD TEST (Walk-Forward)
# ============================================================
print("\n" + "="*70)
print("LONGER OOS / WALK-FORWARD TEST")
print("="*70)

# Test on different OOS periods
wf_periods = [
    ('2026-01-01', '2026-03-31', 'Q1 2026'),
    ('2026-04-01', '2026-06-30', 'Q2 2026'),
    ('2026-07-01', '2026-08-25', 'Q3 2026 (partial)'),
    ('2025-10-01', '2025-12-31', 'Q4 2025'),
    ('2026-01-01', '2026-08-25', 'YTD 2026'),
]

cfg = configs['high_pf_freq']
costs = cost_scenarios['realistic']

print(f"\nConfig: high_pf_freq with realistic costs (spread=0.5, comm=7, slip=0.1)")
print(f"{'Period':<25} {'PF':>6} {'WR':>5} {'DD%':>5} {'N':>5} {'Ret%':>8}")
print(f"{'─'*60}")

for start, end, label in wf_periods:
    mask = (dc['datetime'] >= start) & (dc['datetime'] < end)
    period_data = dc[mask].reset_index(drop=True)
    if len(period_data) < 100:
        continue
    eq, pnls, n = backtest_costs(period_data, cfg,
                                   spread_per_oz=costs['spread'],
                                   commission_per_lot=costs['commission'],
                                   slippage_per_oz=costs['slippage'])
    m = metrics(eq, pnls, n)
    if m:
        print(f"{label:<25} {m['pf']:>6.2f} {m['wr']:>4.0%} {m['mdd']:>4.1f} {m['n']:>5} {m['ret']:>+7.1f}%")

# ============================================================
# MONTHLY BREAKDOWN WITH REALISTIC COSTS
# ============================================================
print("\n" + "="*70)
print("MONTHLY BREAKDOWN (Realistic Costs)")
print("="*70)

eq, pnls, n = backtest_costs(oos_data, cfg,
                               spread_per_oz=costs['spread'],
                               commission_per_lot=costs['commission'],
                               slippage_per_oz=costs['slippage'])

# Monthly from equity curve
dates = oos_data['datetime'].values
months = pd.Series(pd.to_datetime(dates)).dt.to_period('M').unique()
prev_eq = INITIAL_EQUITY
for mo in sorted(months):
    mask = pd.Series(pd.to_datetime(dates)).dt.to_period('M') == mo
    last_idx = mask[mask].index[-1]
    eq_val = eq[last_idx]
    mret = (eq_val - prev_eq) / prev_eq * 100
    print(f"  {mo}: Return={mret:+.2f}% Equity=${eq_val:,.0f}")
    prev_eq = eq_val

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("REALISTIC EXPECTATION SUMMARY")
print("="*70)
print("""
Typical XAUUSD Broker Costs:
  - Spread: 0.3-0.5$ per oz (30-50 pips)
  - Commission: $3.5-7 per lot round trip
  - Slippage: 0.05-0.2$ per oz (varies with volatility)

For 1 standard lot (100 oz):
  - Spread cost: $30-50 per trade
  - Commission: $3.5-7 per trade
  - Slippage: $5-20 per trade
  - Total round trip: $40-77 per lot

For 0.01 lot (micro):
  - Total round trip: $0.40-0.77 per trade

The backtest uses fractional sizing (0.01-2.0 lots), so costs scale.
""")

# Final realistic numbers
print("Realistic OOS Performance (high_pf_freq, realistic costs):")
eq_r, pnls_r, n_r = backtest_costs(oos_data, cfg,
                                     spread_per_oz=0.5,
                                     commission_per_lot=7.0,
                                     slippage_per_oz=0.1)
m_r = metrics(eq_r, pnls_r, n_r)
if m_r:
    print(f"  PF={m_r['pf']:.2f} | WR={m_r['wr']:.0%} | DD={m_r['mdd']:.1f}% | N={m_r['n']} | Ret={m_r['ret']:+.1f}%")

print("\n✅ Cost analysis complete!")
