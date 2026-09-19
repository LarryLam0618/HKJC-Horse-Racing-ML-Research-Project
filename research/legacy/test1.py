import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Step 1: 載入及清洗數據
# ============================================
print("=" * 70)
print("Step 1-5: 準備數據、加入休息天數及訓練 XGBoost 模型")
print("=" * 70)

results = pd.read_csv('results.csv')
races = pd.read_csv('races.csv')
dividends = pd.read_csv('dividends.csv')

# 清洗數值欄位
results['win_odds'] = pd.to_numeric(results['win_odds'], errors='coerce')
results['draw'] = pd.to_numeric(results['draw'], errors='coerce')
results['actual_weight'] = pd.to_numeric(results['actual_weight'], errors='coerce')
results['declared_weight'] = pd.to_numeric(results['declared_weight'], errors='coerce')

# 合併賽事基本資料
race_info = races[['race_date', 'venue', 'race_no', 'race_class', 'distance_m', 'going']]
df = results.merge(race_info, on=['race_date', 'venue', 'race_no'], how='left')
df['won'] = (df['finish_pos'] == 1).astype(int)
df['race_date'] = pd.to_datetime(df['race_date'])

# ============================================
# Step 2: 特徵工程 - 加入「休息天數 (rest_days)」
# ============================================
# 必須先按馬匹 ID 及日期先後次序排序，防止未來數據洩漏
df = df.sort_values(['horse_id', 'race_date']).reset_index(drop=True)
df['rest_days'] = df.groupby('horse_id')['race_date'].diff().dt.days

# 將初出馬（沒有上一場紀錄）的休息天數填補為全體中位數
median_rest = df['rest_days'].median()
df['rest_days'] = df['rest_days'].fillna(median_rest)

# 重新過濾有效賽事資料
df = df.dropna(subset=['win_odds', 'draw', 'race_class', 'distance_m']).copy()
df = df[df['win_odds'] > 1.0].copy()

# 定義特徵清單（把 rest_days 加入數值特徵）
cat_features = ['venue', 'race_class', 'going', 'jockey_name', 'trainer_name']
num_features = ['win_odds', 'draw', 'distance_m', 'actual_weight', 'declared_weight', 'rest_days']
df[num_features] = df[num_features].fillna(df[num_features].median())

# ============================================
# Step 3: 嚴格時間切分與目標編碼 (Target Encoding)
# ============================================
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

# ============================================
# Step 4: 訓練 XGBoost 模型
# ============================================
pos = train_df['won'].sum()
neg = len(train_df) - pos
spw = neg / pos

model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    eval_metric='logloss', early_stopping_rounds=50, random_state=42
)
model.fit(
    train_df[features], train_df['won'], 
    eval_set=[(val_df[features], val_df['won'])], 
    verbose=False
)

test_df['model_prob'] = model.predict_proba(test_df[features])[:, 1]
test_df['saddle'] = pd.to_numeric(test_df['saddle'], errors='coerce').astype(int)
test_df['finish_pos'] = pd.to_numeric(test_df['finish_pos'], errors='coerce')
print("模型訓練完成，開始計算連贏組合 (修正後視偏差及派彩單位)...\n")

# ============================================
# Step 5: 處理派彩數據 (用作真實中獎對照)
# ============================================
dividends['race_date'] = pd.to_datetime(dividends['race_date'])
test_dates = test_df['race_date'].unique()
q_div = dividends[(dividends['pool'].isin(['QUINELLA', 'QUINELLA PLACE'])) & (dividends['race_date'].isin(test_dates))].copy()

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
# Step 6: 無偏差 Walk-Forward 回測 (修正 $10 基準)
# ============================================
print("=" * 70)
print("Step 6: 計算連贏 Edge 及真實 ROI (防後視偏差 & 修正 $10 派彩基準)")
print("=" * 70)

def unbiased_quinella_backtest(test_data, q_div_data, pool_name):
    bets_list = []

    for (date, venue, rno), race_group in test_data.groupby(['race_date', 'venue', 'race_no']):
        valid_group = race_group.dropna(subset=['model_prob', 'finish_pos', 'saddle', 'win_odds'])
        if len(valid_group) < 2: continue

        horses = valid_group[['saddle', 'model_prob', 'win_odds', 'finish_pos']].to_dict('records')

        # 1. 計算市場 WIN 隱含概率 (按場次 Normalize)
        total_implied = sum(1.0 / h['win_odds'] for h in horses)
        for h in horses:
            h['market_win_prob'] = (1.0 / h['win_odds']) / total_implied

        # 2. 獲取該場真實中獎組合
        race_divs = q_div_data[
            (q_div_data['race_date'] == date) & (q_div_data['venue'] == venue) &
            (q_div_data['race_no'] == rno) & (q_div_data['pool'] == pool_name)
        ]

        pos_dict = {h['saddle']: h['finish_pos'] for h in horses}

        # 3. 遍歷所有兩兩組合 (不偷睇賽果)
        for h1, h2 in combinations(horses, 2):
            s1, s2 = h1['saddle'], h2['saddle']

            # 模型連贏概率
            eps = 1e-6
            p_12 = h1['model_prob'] * (h2['model_prob'] / (1 - h1['model_prob'] + eps))
            p_21 = h2['model_prob'] * (h1['model_prob'] / (1 - h2['model_prob'] + eps))
            model_q_prob = p_12 + p_21

            # 市場連贏概率 (用 WIN 賠率推算)
            m_12 = h1['market_win_prob'] * (h2['market_win_prob'] / (1 - h1['market_win_prob'] + eps))
            m_21 = h2['market_win_prob'] * (h1['market_win_prob'] / (1 - h2['market_win_prob'] + eps))
            market_q_prob = m_12 + m_21

            # 推算理論賠率 (扣除 17.5% 抽水)
            implied_div = 0.825 / market_q_prob if market_q_prob > 0 else 999.0

            # Edge = 模型概率 - 市場概率
            edge = model_q_prob - market_q_prob

            # 判斷真實中獎
            won = 0
            actual_div = 0.0
            if pool_name == 'QUINELLA':
                if pos_dict[s1] <= 2 and pos_dict[s2] <= 2:
                    won = 1
            elif pool_name == 'QUINELLA PLACE':
                if pos_dict[s1] <= 3 and pos_dict[s2] <= 3:
                    won = 1

            # 如果中獎，獲取真實派彩；沒中則為 0
            if won == 1:
                div_row = race_divs[((race_divs['h1']==s1) & (race_divs['h2']==s2)) |
                                    ((race_divs['h1']==s2) & (race_divs['h2']==s1))]
                if not div_row.empty:
                    actual_div = div_row.iloc[0]['dividend']

            bets_list.append({
                'edge': edge,
                'model_prob': model_q_prob,
                'market_prob': market_q_prob,
                'implied_div': implied_div,
                'actual_div': actual_div,
                'won': won
            })

    return pd.DataFrame(bets_list)

# --- 測試 QUINELLA ---
print("\n--- QUINELLA (連贏) 真實回測 ---")
quinella_bets = unbiased_quinella_backtest(test_df, q_div, 'QUINELLA')
print(f"總可投注組合: {len(quinella_bets)}")

thresholds = [-0.02, 0.00, 0.01, 0.03, 0.05]
print(f"{'Edge門檻':<12} | {'投注次數':<8} | {'中獎':<6} | {'中獎率':<8} | {'總回報':<12} | {'ROI%':<8}")
print("-" * 70)

for thresh in thresholds:
    bets = quinella_bets[quinella_bets['edge'] > thresh].copy()
    if len(bets) == 0: continue
    
    total_bets = len(bets)
    wins = bets['won'].sum()
    win_rate = (wins / total_bets) * 100
    
    # 【重大修正】：馬會 actual_div 是 $10 派彩基準。
    # 投資 $1 的情況下，贏錢利潤 = (actual_div / 10.0) - 1；輸錢利潤 = -1
    bets['profit'] = bets.apply(lambda r: (r['actual_div'] / 10.0) - 1.0 if r['won'] == 1 else -1.0, axis=1)
    total_profit = bets['profit'].sum()
    roi = (total_profit / total_bets) * 100
    
    print(f"> {thresh:<9.2f} | {total_bets:<8} | {wins:<6} | {win_rate:<8.2f} | {total_profit:<12.2f} | {roi:<+8.2f}")

# --- 測試 QUINELLA PLACE ---
print("\n--- QUINELLA PLACE (位置Q) 真實回測 ---")
q_place_bets = unbiased_quinella_backtest(test_df, q_div, 'QUINELLA PLACE')
print(f"總可投注組合: {len(q_place_bets)}")

print(f"{'Edge門檻':<12} | {'投注次數':<8} | {'中獎':<6} | {'中獎率':<8} | {'總回報':<12} | {'ROI%':<8}")
print("-" * 70)

for thresh in thresholds:
    bets = q_place_bets[q_place_bets['edge'] > thresh].copy()
    if len(bets) == 0: continue
    
    total_bets = len(bets)
    wins = bets['won'].sum()
    win_rate = (wins / total_bets) * 100
    
    # 【重大修正】：同步修正位置 Q 的 $10 派彩基準
    bets['profit'] = bets.apply(lambda r: (r['actual_div'] / 10.0) - 1.0 if r['won'] == 1 else -1.0, axis=1)
    total_profit = bets['profit'].sum()
    roi = (total_profit / total_bets) * 100
    
    print(f"> {thresh:<9.2f} | {total_bets:<8} | {wins:<6} | {win_rate:<8.2f} | {total_profit:<12.2f} | {roi:<+8.2f}")

print("\n" + "=" * 70)
print("真實回測完成！")