import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Step 1: 載入及清洗數據
# ============================================
print("=" * 70)
print("Step 1: 載入及清洗數據")
print("=" * 70)

results = pd.read_csv('results.csv')
races = pd.read_csv('races.csv')

# 基本清洗
results['win_odds'] = pd.to_numeric(results['win_odds'], errors='coerce')
results['draw'] = pd.to_numeric(results['draw'], errors='coerce')
results['actual_weight'] = pd.to_numeric(results['actual_weight'], errors='coerce')
results['declared_weight'] = pd.to_numeric(results['declared_weight'], errors='coerce')

# 合併賽事資料
race_info = races[['race_date', 'venue', 'race_no', 'race_class', 'distance_m', 'going']]
df = results.merge(race_info, on=['race_date', 'venue', 'race_no'], how='left')

# 標記勝出
df['won'] = (df['finish_pos'] == 1).astype(int)
df['race_date'] = pd.to_datetime(df['race_date'])

# 過濾無效賠率同檔位
df = df.dropna(subset=['win_odds', 'draw', 'race_class', 'distance_m']).copy()
df = df[df['win_odds'] > 1.0].copy()

print(f"總有效樣本: {len(df)} 場次")

# ============================================
# Step 2: 準備特徵
# ============================================
print("\n" + "=" * 70)
print("Step 2: 特徵工程")
print("=" * 70)

# 類別特徵
cat_features = ['venue', 'race_class', 'going', 'jockey_name', 'trainer_name']
# 數值特徵
num_features = ['win_odds', 'draw', 'distance_m', 'actual_weight', 'declared_weight']

# 填補數值缺失值
df[num_features] = df[num_features].fillna(df[num_features].median())

print(f"使用特徵: {num_features + [f + '_te' for f in cat_features]}")

# ============================================
# Step 3: 3段式 Walk-Forward 分割
# ============================================
print("\n" + "=" * 70)
print("Step 3: Walk-Forward 時間分割")
print("=" * 70)

# 依時間排序
df = df.sort_values('race_date').reset_index(drop=True)

# 獲取數據的日期範圍來動態分割
min_date = df['race_date'].min()
max_date = df['race_date'].max()

# 假設數據橫跨 2024-25 及 2025-26 賽季
# 我們用日期劃分：Train (最早到 2025-07-31), Val (2025-09-01 到 2025-12-31), Test (2026-01-01 到最後)
train_end = pd.to_datetime('2025-07-31')
val_end = pd.to_datetime('2025-12-31')

train_df = df[df['race_date'] <= train_end].copy()
val_df = df[(df['race_date'] > train_end) & (df['race_date'] <= val_end)].copy()
test_df = df[df['race_date'] > val_end].copy()

print(f"Training Set: {train_df['race_date'].min().date()} 到 {train_df['race_date'].max().date()} ({len(train_df)} 筆)")
print(f"Validation Set: {val_df['race_date'].min().date()} 到 {val_df['race_date'].max().date()} ({len(val_df)} 筆)")
print(f"Test Set: {test_df['race_date'].min().date()} 到 {test_df['race_date'].max().date()} ({len(test_df)} 筆)")

# ============================================
# Step 4: Target Encoding (防 Leakage)
# ============================================
print("\n" + "=" * 70)
print("Step 4: 執行 Target Encoding (僅用 Train 數據計算)")
print("=" * 70)

global_mean = train_df['won'].mean()

for col in cat_features:
    # 用 Train 計算每個類別的勝率
    mapping = train_df.groupby(col)['won'].mean()
    
    # Apply 到 Train, Val, Test
    train_df[col + '_te'] = train_df[col].map(mapping).fillna(global_mean)
    val_df[col + '_te'] = val_df[col].map(mapping).fillna(global_mean)
    test_df[col + '_te'] = test_df[col].map(mapping).fillna(global_mean)

features = num_features + [f + '_te' for f in cat_features]

# ============================================
# Step 5: 訓練 XGBoost 模型
# ============================================
print("\n" + "=" * 70)
print("Step 5: 訓練 XGBoost 模型 (使用 Validation 早期停止)")
print("=" * 70)

# 計算正負樣本比例做 scale_pos_weight (因為賽馬勝率只有 ~9%)
pos = train_df['won'].sum()
neg = len(train_df) - pos
spw = neg / pos

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42
)

model.fit(
    train_df[features], 
    train_df['won'], 
    eval_set=[(val_df[features], val_df['won'])], 
    verbose=10
)

print(f"最佳迭代次數: {model.best_iteration}")

# ============================================
# Step 6: 預測與計算 Edge (Gap)
# ============================================
print("\n" + "=" * 70)
print("Step 6: 測試集預測與 Brier Score 評估")
print("=" * 70)

# 預測機率
test_df['model_prob'] = model.predict_proba(test_df[features])[:, 1]

# 計算市場隱含機率 (按場次 Normalize 解決 Overround)
test_df['raw_market_prob'] = 1.0 / test_df['win_odds']
race_sums = test_df.groupby(['race_date', 'venue', 'race_no'])['raw_market_prob'].transform('sum')
test_df['market_prob'] = test_df['raw_market_prob'] / race_sums

# 計算 Edge: 模型機率 - 市場機率
test_df['edge'] = test_df['model_prob'] - test_df['market_prob']

# Brier Score (越細越好，0.0完美)
brier_model = brier_score_loss(test_df['won'], test_df['model_prob'])
brier_market = brier_score_loss(test_df['won'], test_df['market_prob'])

print(f"市場基準 Brier Score: {brier_market:.4f}")
print(f"XGBoost 模型 Brier Score: {brier_model:.4f}")
if brier_model < brier_market:
    print("結論: 模型預測能力優於市場基準！")
else:
    print("結論: 模型預測能力未及市場基準。")

# ============================================
# Step 7: Edge 訊號回測 (ROI)
# ============================================
print("\n" + "=" * 70)
print("Step 7: Edge 訊號回測 (Test Set)")
print("=" * 70)

# 設定唔同嘅 Edge 門檻
thresholds = [-0.05, 0.0, 0.02, 0.05, 0.10]

print(f"{'Edge門檻':<12} | {'投注場次':<8} | {'勝出':<6} | {'勝率':<8} | {'總回報':<10} | {'ROI%':<8}")
print("-" * 65)

for thresh in thresholds:
    bets = test_df[test_df['edge'] > thresh].copy()
    
    if len(bets) == 0:
        continue
    
    total_bets = len(bets)
    wins = bets['won'].sum()
    win_rate = (wins / total_bets) * 100
    
    # 計算 Profit (每注 $1)
    bets['profit'] = bets.apply(lambda r: r['win_odds'] - 1 if r['won'] == 1 else -1, axis=1)
    total_profit = bets['profit'].sum()
    roi = (total_profit / total_bets) * 100
    
    print(f"> {thresh:<8.2f} | {total_bets:<8} | {wins:<6} | {win_rate:<8.2f} | {total_profit:<10.2f} | {roi:<+8.2f}")

print("\n" + "=" * 70)
print("分析完成！")