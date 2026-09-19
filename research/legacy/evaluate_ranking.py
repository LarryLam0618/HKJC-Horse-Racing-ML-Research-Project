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

features = [
    'win_odds', 'draw', 'actual_weight', 'distance_m',
    'habitual_speed_3', 'habitual_early_pos_pct_3',
    'is_dropping_class', 'is_rising_class', 'class_advantage'
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

print("開始訓練 XGBoost 融合模型...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    eval_metric='logloss', use_label_encoder=False, random_state=42
)
model.fit(X_train, y_train)

# 3. 預測機率
test_df['ml_prob'] = model.predict_proba(X_test)[:, 1]
test_df['implied_prob'] = 1 / test_df['win_odds']

# 4. 計算 Spearman 排序相關係數 (以「場」為單位)
def calculate_spearman(group):
    # 至少要有 3 匹馬才能算有意義的排序
    if len(group) < 3:
        return np.nan, np.nan, np.nan
    
    # 實際名次越小越好，預測機率越大越好，所以用 negative
    sp_market, _ = spearmanr(-group['finish_pos'], group['implied_prob'])
    sp_ml, _ = spearmanr(-group['finish_pos'], group['ml_prob'])
    
    return sp_market, sp_ml, len(group)

print("\n正在計算每場賽事的排序相關係數...")
# 用 run_date + venue + race_index 鎖定唯一賽事
race_groups = test_df.groupby(['run_date', 'venue', 'race_index'])

results = race_groups.apply(calculate_spearman)
results_df = pd.DataFrame(results.tolist(), index=results.index, columns=['SP_Market', 'SP_ML', 'Field_Size']).dropna()

# 5. 整體排序能力對比
avg_sp_market = results_df['SP_Market'].mean()
avg_sp_ml = results_df['SP_ML'].mean()

print("\n" + "="*50)
print("【整體排序能力 (Spearman 相關係數，越高越好)】")
print(f"純賠率排序準確度      : {avg_sp_market:.4f}")
print(f"ML 融合模型排序準確度 : {avg_sp_ml:.4f}")
if avg_sp_ml > avg_sp_market:
    print("✅ 成功！ML 模型在場內排序上優於賠率！")
else:
    print("❌ 賠率排序依然較優。")
print("="*50)

# 6. 子群組分析：尋找市場弱點
print("\n【子群組分析：尋找市場效率弱點】")
# 我們把 test_df 跟 results_df 合併，方便分組
test_df['SP_Market'] = test_df.set_index(['run_date', 'venue', 'race_index']).index.map(results_df['SP_Market'])
test_df['SP_ML'] = test_df.set_index(['run_date', 'venue', 'race_index']).index.map(results_df['SP_ML'])

# 針對場地分析
print("\n--- 按場地分析 ---")
venue_result = test_df.dropna(subset=['SP_Market']).groupby('venue').agg({
    'SP_Market': 'mean',
    'SP_ML': 'mean',
    'race_index': 'count' # 算筆數
}).rename(columns={'race_index': 'Count'})
print(venue_result)

# 針對班次分析 (Class 1-3 為高班，4-5為低班)
test_df['race_class_num'] = pd.to_numeric(test_df['race_class'], errors='coerce')
test_df['class_tier'] = np.where(test_df['race_class_num'] <= 3, 'High_Class', 'Low_Class')

print("\n--- 按班次分析 ---")
class_result = test_df.dropna(subset=['SP_Market']).groupby('class_tier').agg({
    'SP_Market': 'mean',
    'SP_ML': 'mean',
    'race_index': 'count'
}).rename(columns={'race_index': 'Count'})
print(class_result)