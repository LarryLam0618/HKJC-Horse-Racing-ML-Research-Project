import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Step 1-5: 數據處理同 XGBoost 模型訓練
# ============================================
print("=" * 70)
print("Step 1-5: 準備數據及訓練 XGBoost 模型")
print("=" * 70)

results = pd.read_csv('results.csv')
races = pd.read_csv('races.csv')
dividends = pd.read_csv('dividends.csv')

# 清洗 results
results['win_odds'] = pd.to_numeric(results['win_odds'], errors='coerce')
results['draw'] = pd.to_numeric(results['draw'], errors='coerce')
results['actual_weight'] = pd.to_numeric(results['actual_weight'], errors='coerce')
results['declared_weight'] = pd.to_numeric(results['declared_weight'], errors='coerce')

race_info = races[['race_date', 'venue', 'race_no', 'race_class', 'distance_m', 'going']]
df = results.merge(race_info, on=['race_date', 'venue', 'race_no'], how='left')
df['won'] = (df['finish_pos'] == 1).astype(int)
df['race_date'] = pd.to_datetime(df['race_date'])
df = df.dropna(subset=['win_odds', 'draw', 'race_class', 'distance_m']).copy()
df = df[df['win_odds'] > 1.0].copy()

cat_features = ['venue', 'race_class', 'going', 'jockey_name', 'trainer_name']
num_features = ['win_odds', 'draw', 'distance_m', 'actual_weight', 'declared_weight']
df[num_features] = df[num_features].fillna(df[num_features].median())

df = df.sort_values('race_date').reset_index(drop=True)
train_end = pd.to_datetime('2025-07-31')
val_end = pd.to_datetime('2025-12-31')

train_df = df[df['race_date'] <= train_end].copy()
val_df = df[(df['race_date'] > train_end) & (df['race_date'] <= val_end)].copy()
test_df = df[df['race_date'] > val_end].copy()

global_mean = train_df['won'].mean()
for col in cat_features:
    mapping = train_df.groupby(col)['won'].mean()
    train_df[col + '_te'] = train_df[col].map(mapping).fillna(global_mean)
    val_df[col + '_te'] = val_df[col].map(mapping).fillna(global_mean)
    test_df[col + '_te'] = test_df[col].map(mapping).fillna(global_mean)

features = num_features + [f + '_te' for f in cat_features]
pos = train_df['won'].sum()
neg = len(train_df) - pos
spw = neg / pos

model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    eval_metric='logloss', early_stopping_rounds=50, random_state=42
)
model.fit(train_df[features], train_df['won'], eval_set=[(val_df[features], val_df['won'])], verbose=False)

val_df['model_prob'] = model.predict_proba(val_df[features])[:, 1]
test_df['model_prob'] = model.predict_proba(test_df[features])[:, 1]

for d in [val_df, test_df]:
    d['saddle'] = pd.to_numeric(d['saddle'], errors='coerce').astype(int)
    d['finish_pos'] = pd.to_numeric(d['finish_pos'], errors='coerce')

print("模型訓練完成，開始無偏差回測...\n")

# ============================================
# Step 6: 處理派彩數據
# ============================================
dividends['race_date'] = pd.to_datetime(dividends['race_date'])
all_dates = pd.concat([val_df['race_date'], test_df['race_date']]).unique()
q_div = dividends[(dividends['pool'].isin(['QUINELLA', 'QUINELLA PLACE'])) & (dividends['race_date'].isin(all_dates))].copy()

def parse_comb(s):
    parts = str(s).split(',')
    if len(parts) == 2:
        try: return int(parts[0].strip()), int(parts[1].strip())
        except: return None, None
    return None, None

q_div['h1'] = q_div['combination'].apply(lambda x: parse_comb(x)[0])
q_div['h2'] = q_div['combination'].apply(lambda x: parse_comb(x)[1])
q_div = q_div.dropna(subset=['h1', 'h2'])

# ============================================
# Step 7: Harville Formula 計算函數
# ============================================
def calc_quinella_prob(idx_a, idx_b, probs):
    """計算 QUINELLA (連贏) 機率"""
    p_a = probs[idx_a]
    p_b = probs[idx_b]
    eps = 1e-6
    p_12 = p_a * (p_b / (1 - p_a + eps))
    p_21 = p_b * (p_a / (1 - p_b + eps))
    return p_12 + p_21

def calc_qp_prob(idx_a, idx_b, probs):
    """計算 QUINELLA PLACE (位置Q) 機率: 兩匹馬都跑入前 3 名"""
    p_a = probs[idx_a]
    p_b = probs[idx_b]
    total_prob = 0.0
    eps = 1e-6
    
    # 兩匹馬都在前3，代表第3個名次一定是另一匹馬 k
    # 我哋要 Sum over 所有其他馬匹 k，計算 6 種排列嘅 Harville 機率
    for idx_k, p_k in enumerate(probs):
        if idx_k == idx_a or idx_k == idx_b: continue
        
        # (A, B, k)
        total_prob += p_a * (p_b / (1 - p_a + eps)) * (p_k / (1 - p_a - p_b + eps))
        # (A, k, B)
        total_prob += p_a * (p_k / (1 - p_a + eps)) * (p_b / (1 - p_a - p_k + eps))
        # (B, A, k)
        total_prob += p_b * (p_a / (1 - p_b + eps)) * (p_k / (1 - p_b - p_a + eps))
        # (B, k, A)
        total_prob += p_b * (p_k / (1 - p_b + eps)) * (p_a / (1 - p_b - p_k + eps))
        # (k, A, B)
        total_prob += p_k * (p_a / (1 - p_k + eps)) * (p_b / (1 - p_k - p_a + eps))
        # (k, B, A)
        total_prob += p_k * (p_b / (1 - p_k + eps)) * (p_a / (1 - p_k - p_b + eps))
        
    return total_prob

def calculate_quinella_edges(data_df, q_div_data, pool_name):
    bets_list = []
    
    for (date, venue, rno), race_group in data_df.groupby(['race_date', 'venue', 'race_no']):
        valid_group = race_group.dropna(subset=['model_prob', 'finish_pos', 'saddle', 'win_odds'])
        if len(valid_group) < 3: continue # QP 至少要有3匹馬
            
        horses = valid_group[['saddle', 'model_prob', 'win_odds', 'finish_pos']].to_dict('records')
        
        # 計算市場 WIN 隱含概率
        total_implied = sum(1.0 / h['win_odds'] for h in horses)
        for h in horses:
            h['market_win_prob'] = (1.0 / h['win_odds']) / total_implied
            
        race_divs = q_div_data[
            (q_div_data['race_date'] == date) & (q_div_data['venue'] == venue) & 
            (q_div_data['race_no'] == rno) & (q_div_data['pool'] == pool_name)
        ]
        
        pos_dict = {h['saddle']: h['finish_pos'] for h in horses}
        
        # 準備機率列表俾 Harville function 用
        model_probs_list = [h['model_prob'] for h in horses]
        market_probs_list = [h['market_win_prob'] for h in horses]
        
        for i, j in combinations(range(len(horses)), 2):
            h1 = horses[i]
            h2 = horses[j]
            s1, s2 = h1['saddle'], h2['saddle']
            
            if pool_name == 'QUINELLA':
                model_q_prob = calc_quinella_prob(i, j, model_probs_list)
                market_q_prob = calc_quinella_prob(i, j, market_probs_list)
            else: # QUINELLA PLACE
                model_q_prob = calc_qp_prob(i, j, model_probs_list)
                market_q_prob = calc_qp_prob(i, j, market_probs_list)
                
            implied_div = 0.825 / market_q_prob if market_q_prob > 0 else 999.0
            edge = model_q_prob - market_q_prob
            
            won = 0
            actual_div = 0.0
            if pool_name == 'QUINELLA':
                if pos_dict[s1] <= 2 and pos_dict[s2] <= 2: won = 1
            elif pool_name == 'QUINELLA PLACE':
                if pos_dict[s1] <= 3 and pos_dict[s2] <= 3: won = 1
                    
            if won == 1:
                div_row = race_divs[((race_divs['h1']==s1) & (race_divs['h2']==s2)) | 
                                    ((race_divs['h1']==s2) & (race_divs['h2']==s1))]
                if not div_row.empty:
                    actual_div = div_row.iloc[0]['dividend']
            
            bets_list.append({
                'race_id': f"{date}_{venue}_{rno}",
                'edge': edge,
                'implied_div': implied_div,
                'actual_div_10': actual_div,
                'won': won
            })
            
    return pd.DataFrame(bets_list)

def analyze_pool(pool_name, val_df, test_df, q_div):
    print("=" * 70)
    print(f"分析 {pool_name}")
    print("=" * 70)
    
    val_bets = calculate_quinella_edges(val_df, q_div, pool_name)
    test_bets = calculate_quinella_edges(test_df, q_div, pool_name)
    
    # === 1. 防過擬合 Threshold 揀選 (強制最少 200 注) ===
    thresholds = np.arange(-0.05, 0.06, 0.01)
    best_thresh = None
    best_roi = -999
    min_bets_req = 200
    
    print(f"驗證集 尋找最佳 Threshold (強制最少 {min_bets_req} 注)...")
    for thresh in thresholds:
        bets = val_bets[val_bets['edge'] > thresh].copy()
        if len(bets) < min_bets_req: continue
        
        bets['profit'] = bets.apply(lambda r: (r['actual_div_10'] / 10.0) - 1 if r['won'] == 1 else -1, axis=1)
        roi = (bets['profit'].sum() / len(bets)) * 100
        
        if roi > best_roi:
            best_roi = roi
            best_thresh = thresh
            
    if best_thresh is None:
        print(f"-> 驗證集冇任何 Threshold 達到 {min_bets_req} 注要求，策略不可信。")
        return
        
    print(f"-> 驗證集最佳 Threshold: > {best_thresh:.2f} (ROI: {best_roi:+.2f}%)")
    
    # === 2. 鎖定 Threshold 喺 Test Set 驗證 ===
    print("\n測試集 驗證結果 (鎖定 Threshold):")
    test_selected = test_bets[test_bets['edge'] > best_thresh].copy()
    
    if len(test_selected) == 0:
        print("測試集冇符合條件嘅投注。")
        return
        
    total_bets = len(test_selected)
    wins = test_selected['won'].sum()
    win_rate = (wins / total_bets) * 100
    
    test_selected['profit'] = test_selected.apply(lambda r: (r['actual_div_10'] / 10.0) - 1 if r['won'] == 1 else -1, axis=1)
    total_profit = test_selected['profit'].sum()
    bet_level_roi = (total_profit / total_bets) * 100
    
    print(f"總投注次數: {total_bets}")
    print(f"總中獎次數: {wins}")
    print(f"中獎率:     {win_rate:.2f}%")
    print(f"總回報:     {total_profit:.2f}")
    print(f"逐注 ROI:   {bet_level_roi:+.2f}%")
    
    # === 3. 按場次 彙總穩健性檢查 (修正計算邏輯) ===
    race_level = test_selected.groupby('race_id').agg(
        bets=('won', 'count'),
        race_profit=('profit', 'sum')
    ).reset_index()
    
    # 修正: 每場 ROI = (該場總 Profit / 該場總注數) * 100，再求平均
    race_level['race_roi'] = (race_level['race_profit'] / race_level['bets']) * 100
    race_level_roi = race_level['race_roi'].mean()
    
    print(f"場次 ROI:   {race_level_roi:+.2f}% (共 {len(race_level)} 場)")
    
    # === 4. Payout Factor 健全性檢查 ===
    print("\n理論賠率 vs 真實賠率 健全性檢查:")
    won_bets = test_selected[test_selected['won'] == 1].copy()
    if len(won_bets) > 0:
        won_bets['actual_div_1'] = won_bets['actual_div_10'] / 10.0
        ratio = (won_bets['actual_div_1'] / won_bets['implied_div']).median()
        print(f"真實派彩 (每$1) 中位數: {won_bets['actual_div_1'].median():.2f}")
        print(f"理論賠率 中位數:        {won_bets['implied_div'].median():.2f}")
        print(f"兩者比率 (真實/理論):   {ratio:.2f}")
        
        if ratio < 0.7 or ratio > 1.0:
            print("⚠️ 警告: 比率嚴重偏離 0.825，市場抽水假設可能錯誤，Edge 計算可能有系統性偏差！")
        else:
            print("✅ 比率接近 0.825，假設合理。")
            
    print("-" * 70 + "\n")

analyze_pool('QUINELLA', val_df, test_df, q_div)
analyze_pool('QUINELLA PLACE', val_df, test_df, q_div)