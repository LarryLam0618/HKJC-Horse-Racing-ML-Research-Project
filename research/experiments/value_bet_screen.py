import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 設定
# ==========================================
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)
BASE_DATA_PATH = Path("training_set_safe.csv")

# 自動偵測可能嘅預測檔案
PRED_CANDIDATES = [
    OUT_DIR / "final_model_predictions.csv",
    OUT_DIR / "lightgbm_native_predictions_oof_eval.csv",
    OUT_DIR / "baseline_predictions_oof_calibrated.csv"
]

def resolve_pred_path():
    for p in PRED_CANDIDATES:
        if p.exists():
            print(f"✅ 使用預測檔案: {p}")
            return p
    raise FileNotFoundError(f"❌ 搵唔到預測檔案，請檢查 output/ 資料夾入面有冇 CSV 檔案。")

def load_and_merge_odds(df):
    # 如果冇 win_odds，就去 training_set_safe.csv 度 merge 返嚟
    if 'win_odds' not in df.columns:
        print(f"⚠️ 預測檔案缺少 win_odds，正在從 {BASE_DATA_PATH} 合併...")
        if not BASE_DATA_PATH.exists():
            raise FileNotFoundError(f"❌ 找不到 {BASE_DATA_PATH}")
            
        base_df = pd.read_csv(BASE_DATA_PATH, usecols=['race_date', 'venue', 'race_no', 'horse_id', 'win_odds'])
        df = df.merge(base_df, on=['race_date', 'venue', 'race_no', 'horse_id'], how='left')
        print("✅ win_odds 合併成功！")
        
    return df

def calculate_value_bets(df):
    print("📊 計算期望值 同投資邊際...")
    
    # 確保 win_odds 係數值
    df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
    
    # 自動偵測機率欄位名稱
    prob_col = None
    for col in ['calibrated_prob', 'pred_win_prob_platt_oof', 'pred_win_prob']:
        if col in df.columns:
            prob_col = col
            break
            
    if prob_col is None:
        raise ValueError("❌ 找不到機率預測欄位 (calibrated_prob / pred_win_prob_platt_oof)")
        
    print(f"📌 使用機率欄位: {prob_col}")
    df = df.dropna(subset=['win_odds', prob_col]).copy()
    
    # 1. 計算市場隱含勝率 (需扣除莊家抽水，HKJC 大約 17.5%)
    df['market_implied_prob'] = 1 / df['win_odds']
    
    # 2. 計算期望值 (EV)
    df['expected_value'] = (df[prob_col] * df['win_odds']) - 1
    
    # 3. 計算模型 Edge
    df['model_edge'] = df[prob_col] - df['market_implied_prob']
    
    # 統一改名方便後續處理
    df = df.rename(columns={prob_col: 'model_prob'})
    
    return df

def screen_and_report(df):
    # 只篩選 EV > 0% 嘅馬
    value_bets = df[df['expected_value'] > 0].copy()
    
    # 按 EV 高低排序
    value_bets = value_bets.sort_values(by='expected_value', ascending=False)
    
    print("\n" + "=" * 70)
    print("💰 價值投注篩選報告 (基於 LightGBM 校準機率)")
    print("=" * 70)
    print(f"總共分析場次: {df.groupby(['race_date', 'venue', 'race_no']).ngroups} 場")
    print(f"總共馬匹記錄: {len(df)} 匹")
    print(f"篩選出有正 EV 嘅馬匹: {len(value_bets)} 匹 ({len(value_bets)/len(df):.2%})")
    
    # 統計實際命中情況
    # 偵測目標欄位
    target_col = 'win_label' if 'win_label' in value_bets.columns else 'won'
    
    if target_col in value_bets.columns:
        hit_rate = value_bets[target_col].mean()
        # 假設每注 1 元
        total_return = (value_bets[target_col] * value_bets['win_odds']).sum()
        total_stake = len(value_bets) * 1.0  
        roi = (total_return - total_stake) / total_stake
        
        print("\n📈 盲買所有正 EV 馬匹嘅歷史回測表現:")
        print(f"  命中率       : {hit_rate:.2%}")
        print(f"  總注數       : {int(total_stake)} 注")
        print(f"  總回報       : {total_return:.2f} 元")
        print(f"  投資回報率 ROI: {roi:.2%}")
        
    print("=" * 70)
    
    # 儲存結果
    value_bets.to_csv(OUT_DIR / "value_bets_screen.csv", index=False)
    print(f"\n✅ 價值投注清單已儲存至: {OUT_DIR / 'value_bets_screen.csv'}")
    
    # 顯示 Top 10 最高 EV 嘅馬匹
    print("\n🌟 Top 10 最高 EV 嘅歷史馬匹:")
    display_cols = ['race_date', 'race_no', 'horse_id', 'win_odds', 'model_prob', 'expected_value', target_col]
    display_cols = [c for c in display_cols if c in value_bets.columns]
    print(value_bets[display_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    pred_path = resolve_pred_path()
    print(f"📂 讀取預測數據: {pred_path} ...")
    df = pd.read_csv(pred_path)
    
    # 自動補齊 win_odds
    df = load_and_merge_odds(df)
    
    df = calculate_value_bets(df)
    screen_and_report(df)