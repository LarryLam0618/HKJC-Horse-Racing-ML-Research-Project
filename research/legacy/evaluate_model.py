import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

print("正在載入特徵資料 (包含體重變化)...")
df = pd.read_csv('horse_form_features_ready.csv')

# 1. 清理與定義
df = df.dropna(subset=['run_date', 'finish_pos', 'win_odds']).copy()
df['run_date'] = pd.to_datetime(df['run_date'])
df['finish_pos'] = pd.to_numeric(df['finish_pos'], errors='coerce')
df = df.dropna(subset=['finish_pos'])

df['is_win'] = (df['finish_pos'] == 1).astype(int)

# 這次：保留 win_odds，並加入體重變化特徵
features = [
    'win_odds', 'draw', 'actual_weight', 'distance_m',  # 市場與靜態
    'habitual_speed_3', 'habitual_early_pos_pct_3',      # 動態速度與跑法
    'is_dropping_class', 'is_rising_class', 'class_advantage', # 班次變動
    'weight_diff', 'weight_diff_pct', 'weight_diff_3_race_trend' # 體重變化 (新增!)
]

X = df[features]
y = df['is_win']

# 2. 按時間切分
split_date = df['run_date'].quantile(0.8)
X_train = X[df['run_date'] < split_date]
y_train = y[df['run_date'] < split_date]
test_df = df[df['run_date'] >= split_date].copy()
X_test = test_df[features]
y_test = test_df['is_win']

print(f"訓練集時間範圍: {df[df['run_date'] < split_date]['run_date'].min().date()} 到 {df[df['run_date'] < split_date]['run_date'].max().date()}")
print(f"測試集時間範圍: {df[df['run_date'] >= split_date]['run_date'].min().date()} 到 {df['run_date'].max().date()}")

# 3. 訓練融合模型 (不加 scale_pos_weight 以保持機率校準)
print("\n開始訓練 XGBoost 融合模型 (自製特徵 + 賠率 + 體重)...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    eval_metric='logloss', use_label_encoder=False, random_state=42
)
model.fit(X_train, y_train)

# 4. 評估
y_prob = model.predict_proba(X_test)[:, 1]
valid_mask = X_test['win_odds'].notna()
y_test_valid = y_test[valid_mask]
y_prob_valid = y_prob[valid_mask]
win_odds_valid = X_test['win_odds'][valid_mask]

brier_model = brier_score_loss(y_test_valid, y_prob_valid)
brier_market = brier_score_loss(y_test_valid, 1 / win_odds_valid)

print("\n" + "="*50)
print("【Brier Score 對比 (融合模型 - 含體重特徵)】")
print(f"純賠率基準      : {brier_market:.4f}")
print(f"ML 融合模型     : {brier_model:.4f}")

if brier_model < brier_market:
    print("✅ 成功突破！ML 模型擊敗了市場效率天花板！")
    print("-> 體重變化特徵可能成功修正了賠率的誤差！")
else:
    print("⚠️ 結果：融合模型仍未超越賠率。")
    print("-> 體重變化雖有預測力，但可能已被賠率隱含吸收。")
print("="*50)

# 5. 特徵重要性
print("\n【特徵重要性 (融合模型)】")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_df.to_string(index=False))