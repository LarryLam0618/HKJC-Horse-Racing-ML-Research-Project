import pandas as pd
import numpy as np

# ============================================
# Step 1: 載入數據
# ============================================
dividends = pd.read_csv('dividends.csv')
races = pd.read_csv('races.csv')
results = pd.read_csv('results.csv')

# ============================================
# Step 2: 數據清洗 & 合併
# ============================================
# 過濾無效賠率同檔位
results['win_odds'] = pd.to_numeric(results['win_odds'], errors='coerce')
results['draw'] = pd.to_numeric(results['draw'], errors='coerce')
results = results.dropna(subset=['win_odds', 'draw'])
results = results[results['win_odds'] > 1.0] # 賠率一定要大於1

# 從 races 提取賽事級別信息
race_info = races[['race_date', 'venue', 'race_no', 'race_class', 'distance_m', 'going', 'course']].copy()

# 合併到 results
df = results.merge(race_info, on=['race_date', 'venue', 'race_no'], how='left')

# 標記勝出馬
df['won'] = (df['finish_pos'] == 1).astype(int)

# ============================================
# Step 3: 計算隱含概率 (僅作參考)
# ============================================
df['implied_prob'] = 1.0 / df['win_odds']

# ============================================
# Step 4: Draw Bias 探索性分析
# ============================================
print("=" * 70)
print("STEP 4: Draw Bias 探索性分析 (HV only)")
print("=" * 70)

hv_data = df[df['venue'] == 'HV'].copy()

draw_1_4 = hv_data[hv_data['draw'].isin([1, 2, 3, 4])]
draw_5_plus = hv_data[hv_data['draw'] >= 5]

print(f"\n--- Draw 1-4 vs Draw 5+ ---")
print(f"Draw 1-4:  {len(draw_1_4):5d} runs | {draw_1_4['won'].sum():4d} wins | {draw_1_4['won'].mean()*100:.2f}% win rate")
print(f"Draw 5+:   {len(draw_5_plus):5d} runs | {draw_5_plus['won'].sum():4d} wins | {draw_5_plus['won'].mean()*100:.2f}% win rate")

# ============================================
# Step 5: Walk-Forward Test (改用賠率區間尋找 Value)
# ============================================
print("\n" + "=" * 70)
print("STEP 5: Walk-Forward ROI Test (利用賠率同檔位尋找價值)")
print("=" * 70)

df['race_date_dt'] = pd.to_datetime(df['race_date'])
df = df.sort_values(['race_date_dt', 'venue', 'race_no']).reset_index(drop=True)

strategies = {
    'A: All HV (Baseline)': {
        'venue_filter': ['HV'], 'draw_filter': None, 'odds_filter': None,
    },
    'B: HV + Draw 1-4': {
        'venue_filter': ['HV'], 'draw_filter': [1, 2, 3, 4], 'odds_filter': None,
    },
    'C: HV + Draw 1-4 + Odds 3.0-8.0 (熱門好檔)': {
        'venue_filter': ['HV'], 'draw_filter': [1, 2, 3, 4], 'odds_filter': (3.0, 8.0),
    },
    'D: HV + Draw 1-4 + Odds 8.0-15.0 (中冷門好檔)': {
        'venue_filter': ['HV'], 'draw_filter': [1, 2, 3, 4], 'odds_filter': (8.0, 15.0),
    },
    'E: HV + Draw 1-4 + Odds 15.0+ (大冷門好檔)': {
        'venue_filter': ['HV'], 'draw_filter': [1, 2, 3, 4], 'odds_filter': (15.0, 999.0),
    },
}

all_results = []

for strategy_name, config in strategies.items():
    mask = df['venue'].isin(config['venue_filter'])
    
    if config['draw_filter'] is not None:
        mask = mask & (df['draw'].isin(config['draw_filter']))
    
    if config['odds_filter'] is not None:
        mask = mask & (df['win_odds'] >= config['odds_filter'][0]) & (df['win_odds'] < config['odds_filter'][1])
    
    selected = df[mask].copy()
    
    if len(selected) == 0:
        continue
    
    # 計算投注結果 (每注 $1)
    selected['profit'] = selected.apply(
        lambda row: row['win_odds'] - 1 if row['won'] == 1 else -1, axis=1
    )
    
    total_bets = len(selected)
    total_wins = selected['won'].sum()
    total_staked = total_bets * 1.0
    total_profit = selected['profit'].sum()
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0
    avg_odds = selected['win_odds'].mean()
    
    all_results.append({
        'Strategy': strategy_name,
        'Bets': total_bets,
        'Wins': total_wins,
        'Win%': round(win_rate, 2),
        'AvgOdds': round(avg_odds, 2),
        'Profit': round(total_profit, 2),
        'ROI%': round(roi, 2),
    })

summary_df = pd.DataFrame(all_results)
print("\n策略對比結果:")
print(summary_df.to_string(index=False))

# ============================================
# Step 6 & 7: 分班次 / 分路程 Draw Bias
# ============================================
print("\n" + "=" * 70)
print("STEP 6 & 7: 分班次同路程 Draw Bias 分析 (HV)")
print("=" * 70)

for feature in ['race_class', 'distance_m']:
    print(f"\n--- 按 {feature} 分析 ---")
    for val in sorted(hv_data[feature].dropna().unique()):
        val_data = hv_data[hv_data[feature] == val]
        d14 = val_data[val_data['draw'].isin([1, 2, 3, 4])]
        d5p = val_data[val_data['draw'] >= 5]
        
        if len(d14) > 20 and len(d5p) > 20:
            d14_roi = ((d14['won'] * (d14['win_odds'] - 1) - (~d14['won'].astype(bool) * 1)).sum() / len(d14)) * 100
            d5p_roi = ((d5p['won'] * (d5p['win_odds'] - 1) - (~d5p['won'].astype(bool) * 1)).sum() / len(d5p)) * 100
            print(f"  {val}: D1-4 WR={d14['won'].mean()*100:5.2f}% (ROI: {d14_roi:+6.2f}%) | D5+ WR={d5p['won'].mean()*100:5.2f}% (ROI: {d5p_roi:+6.2f}%)")

# ============================================
# Step 8: 賠率分位數 × Draw 分析
# ============================================
print("\n" + "=" * 70)
print("STEP 8: 賠率分位數 × Draw 分析 (HV)")
print("=" * 70)

hv_valid = hv_data.dropna(subset=['win_odds']).copy()
# 用賠率分位數代替之前的 underpriced_value
hv_valid['odds_quartile'] = pd.qcut(hv_valid['win_odds'], 4, labels=['Q1(熱門)', 'Q2', 'Q3', 'Q4(大冷門)'])

pivot = hv_valid.groupby(['odds_quartile', pd.cut(hv_valid['draw'], [0, 4, 14])], observed=True).agg(
    bets=('won', 'count'),
    wins=('won', 'sum'),
    win_rate=('won', 'mean'),
    avg_odds=('win_odds', 'mean'),
).reset_index()

pivot['ROI%'] = pivot.apply(
    lambda r: ((r['wins'] * (r['avg_odds'] - 1) - (r['bets'] - r['wins'])) / r['bets']) * 100 if r['bets'] > 0 else 0,
    axis=1
)

print(pivot.to_string(index=False))

# ============================================
# Step 9: 最終最佳組合測試
# ============================================
print("\n" + "=" * 70)
print("STEP 9: 最佳策略組合 ROI 測試")
print("=" * 70)

best_combos = [
    ('HV + Draw 1-4', 
     df[(df['venue']=='HV') & (df['draw'].isin([1,2,3,4]))]),
    ('HV + Draw 1-4 + Class 4-5',
     df[(df['venue']=='HV') & (df['draw'].isin([1,2,3,4])) & (df['race_class'].isin(['Class 4', 'Class 5']))]),
    ('HV + Draw 1-4 + 1200m',
     df[(df['venue']=='HV') & (df['draw'].isin([1,2,3,4])) & (df['distance_m']==1200)]),
    ('HV + Draw 1-4 + 1000m',
     df[(df['venue']=='HV') & (df['draw'].isin([1,2,3,4])) & (df['distance_m']==1000)]),
    ('HV + Draw 1-4 + 1000/1200m + Odds 5-15',
     df[(df['venue']=='HV') & (df['draw'].isin([1,2,3,4])) & (df['distance_m'].isin([1000, 1200])) & (df['win_odds']>=5) & (df['win_odds']<15)]),
]

for name, data in best_combos:
    if len(data) == 0:
        continue
    bets = len(data)
    wins = data['won'].sum()
    profit = data.apply(lambda r: r['win_odds'] - 1 if r['won'] == 1 else -1, axis=1).sum()
    roi = (profit / bets) * 100
    print(f"{name:55s} | {bets:4d} bets | {wins:3d} wins | {wins/bets*100:5.2f}% WR | ROI: {roi:+.2f}%")

print("\n" + "=" * 70)
print("分析完成！")