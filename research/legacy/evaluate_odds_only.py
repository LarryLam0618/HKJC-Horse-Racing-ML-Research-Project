import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

print("正在載入特徵資料...")
df = pd.read_csv('horse_form_features_ready.csv')

# 1. 清理與定義
df = df.dropna(subset=['run_date', 'finish_pos', 'win_odds']).copy()
df['run_date'] = pd.to_datetime(df['run_date'])
df['finish_pos'] = pd.to_numeric(df['finish_pos'], errors='coerce')
df = df.dropna(subset=['finish_pos'])

df['is_win'] = (df['finish_pos'] == 1).astype(int)

# 【對照組設定：只放 win_odds】
features = ['win_odds']

X = df[features]
y = df['is_win']

# 2. 按時間切分
split_date = df['run_date'].quantile(0.8)
X_train = X[df['run_date'] < split_date]
y_train = y[df['run_date'] < split_date]
test_df = df[df['run_date'] >= split_date].copy()
X_test = test_df[features]
y_test = test_df['is_win']

print("開始訓練純賠率 XGBoost 模型...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    eval_metric='logloss', use_label_encoder=False, random_state=42
)
model.fit(X_train, y_train)

# 3. 預測機率
test_df['ml_prob'] = model.predict_proba(X_test)[:, 1]
test_df['implied_prob'] = 1 / test_df['win_odds']

# 4. 計算 Spearman 排序相關係數
def calculate_spearman(group):
    if len(group) < 3: return np.nan, np.nan, np.nan
    sp_market, _ = spearmanr(-group['finish_pos'], group['implied_prob'])
    sp_ml, _ = spearmanr(-group['finish_pos'], group['ml_prob'])
    return sp_market, sp_ml, len(group)

print("正在計算每場賽事的排序相關係數...")
race_groups = test_df.groupby(['run_date', 'venue', 'race_index'])
results = race_groups.apply(calculate_spearman)
results_df = pd.DataFrame(results.tolist(), index=results.index, columns=['SP_Market', 'SP_ML', 'Field_Size']).dropna()

avg_sp_market = results_df['SP_Market'].mean()
avg_sp_ml = results_df['SP_ML'].mean()

print("\n" + "="*50)
print("【最終對照：排序能力 (Spearman)】")
print(f"純賠率直接排序 (1/odds)      : {avg_sp_market:.4f}")
print(f"純賠率 XGBoost 模型排序      : {avg_sp_ml:.4f}")
print(f"融合特徵 XGBoost 模型 (上一輪): 0.4774")
print("="*50)

if avg_sp_ml > 0.4774:
    print("✅ 結論確認：移除自製特徵後，純賠率模型表現更好。自製特徵確實為噪音。")
else:
    print("⚠️ 結論反轉：純賠率模型不如融合模型，特徵可能仍有微小邊際效用。")