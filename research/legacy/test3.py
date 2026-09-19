import pandas as pd
import numpy as np

def extract_running_style(df):
    print("正在處理走位資料 (修正 NaN 邏輯)...")
    
    # 1. 處理 running_position_raw
    df['run_pos_list'] = df['running_position_raw'].astype(str).str.split()
    
    # 2. 提取早段絕對位置
    def get_avg_pos(lst, start_ratio, end_ratio):
        try:
            lst = [int(x) for x in lst]
            length = len(lst)
            if length == 0: return np.nan
            start_idx = int(length * start_ratio)
            end_idx = int(length * end_ratio)
            segment = lst[start_idx:end_idx]
            return np.mean(segment) if segment else np.nan
        except:
            return np.nan

    df['early_pos'] = df['run_pos_list'].apply(lambda x: get_avg_pos(x, 0, 0.33))

    df['run_date'] = pd.to_datetime(df['run_date'], errors='coerce')
    
    # 【終極修正：使用複合鍵鎖定唯一賽事】
    df['field_size'] = df.groupby(['run_date', 'venue', 'race_index'])['horse_id'].transform('count')
    
    # 計算早段位置的「相對百分位」
    df['early_pos_pct'] = df['early_pos'] / df['field_size']

    # 3. 計算馬匹速度 (米/秒)
    df['finish_time_s'] = pd.to_numeric(df['finish_time_s'], errors='coerce')
    df['avg_speed_mps'] = df['distance_m'] / df['finish_time_s']

    # 【Bug 1 修正：使用 loc 條件賦值，完美處理 NaN 與字串混合】
    df['running_style'] = 'P'
    df.loc[df['early_pos_pct'] >= 0.6, 'running_style'] = 'S'
    df.loc[df['early_pos_pct'] <= 0.3, 'running_style'] = 'E'
    df.loc[df['early_pos_pct'].isna(), 'running_style'] = np.nan

    # 4. 計算過去 3 場的「慣常跑法」與「平均速度」
    print("正在計算歷史走位特徵...")
    df = df.sort_values(by=['horse_id', 'run_date'], na_position='last').reset_index(drop=True)
    
    df['date_missing_flag'] = df['run_date'].isna().astype(int)
    
    df['habitual_early_pos_pct_3'] = df.groupby('horse_id')['early_pos_pct'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['habitual_speed_3'] = df.groupby('horse_id')['avg_speed_mps'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    df = df.drop(columns=['run_pos_list'])
    return df

def extract_class_movement(df):
    print("正在計算班次變動與往績水準...")
    df['race_class'] = pd.to_numeric(df['race_class'], errors='coerce')
    
    df['prev_class'] = df.groupby('horse_id')['race_class'].shift(1)
    df['class_movement'] = df['race_class'] - df['prev_class']
    
    df['is_dropping_class'] = np.where(df['class_movement'] > 0, 1, 0)
    df['is_rising_class'] = np.where(df['class_movement'] < 0, 1, 0)
    df['is_same_class'] = np.where(df['class_movement'] == 0, 1, 0)
    
    df['avg_prev_class_3'] = df.groupby('horse_id')['race_class'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['class_advantage'] = df['race_class'] - df['avg_prev_class_3']
    
    return df

def extract_weight_features(df):
    print("正在計算馬匹體重變化特徵...")
    # 確保資料按馬匹和比賽日期排序
    df = df.sort_values(by=['horse_id', 'run_date'], na_position='last').reset_index(drop=True)
    
    # 確保 declared_weight 是數值型態
    df['declared_weight'] = pd.to_numeric(df['declared_weight'], errors='coerce')
    
    # 1. 取得上一場比賽的體重 (使用 shift 避免 leakage)
    df['prev_declared_weight'] = df.groupby('horse_id')['declared_weight'].shift(1)
    
    # 2. 計算絕對體重變化 (今場體重 - 上場體重)
    # 負數代表減重 (狀態上升)，正數代表增重 (狀態可能下滑)
    df['weight_diff'] = df['declared_weight'] - df['prev_declared_weight']
    
    # 3. 計算體重變化百分比 (對於不同體型的馬匹更公平，防止除以0)
    df['weight_diff_pct'] = df['weight_diff'] / df['prev_declared_weight'].replace(0, np.nan)
    
    # 4. 計算過去 3 場的體重變化總和，判斷長期趨勢
    df['weight_diff_3_race_trend'] = df.groupby('horse_id')['weight_diff'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    
    return df

if __name__ == "__main__":
    try:
        file_path = 'horse_form_final.csv'
        print(f"正在載入 CSV 資料 ({file_path})...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # ==========================================
        # 第一步驗證：確認 run_date 缺失率
        # ==========================================
        df['run_date'] = pd.to_datetime(df['run_date'], errors='coerce')
        missing_count = df['run_date'].isna().sum()
        total_count = len(df)
        
        if missing_count > 0:
            print(f"⚠️ 提示：仍有 {missing_count/total_count:.1%} 的日期缺失。")
            print("系統將自動跳過缺失日期的賽事 (field_size 標記為 NaN)，不會影響有日期資料的準確度。")

        # 確保 finish_pos 是數值，以利後續 Sanity Check
        df['finish_pos'] = pd.to_numeric(df['finish_pos'], errors='coerce')

        # 1. 提取走位與速度特徵
        df = extract_running_style(df)
        
        # 2. 提取班次變動特徵
        df = extract_class_movement(df)
        
        # 3. 提取體重變化特徵
        df = extract_weight_features(df)
        
        # ==========================================
        # Sanity Check 區域
        # ==========================================
        print("\n" + "="*50)
        print("【檢查 1：跑法分佈 (已修正 NaN 污染)】")
        valid_style_df = df.dropna(subset=['running_style'])
        print(valid_style_df['running_style'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
        print(f"(註：共有 {df['running_style'].isna().sum()} 筆資料因缺日期被標記為 NaN)")
        
        print("\n" + "="*50)
        print("【檢查 2：不同跑法在 HV (快活谷) 的平均名次】")
        hv_df = valid_style_df[valid_style_df['venue'] == 'HV']
        if not hv_df.empty:
            style_result = hv_df.groupby('running_style')['finish_pos'].mean().sort_values()
            print(style_result)
            
        print("\n" + "="*50)
        print("【檢查 3：班次變動三組對比 (升/降/同班)】")
        valid_class_df = df.dropna(subset=['class_movement'])
        class_result = valid_class_df.groupby(
            ['is_dropping_class', 'is_rising_class', 'is_same_class']
        )['finish_pos'].agg(['mean', 'count'])
        print(class_result)
        print("-> 預期：降班(1,0,0)名次最好，升班(0,1,0)名次最差")
        
        print("\n" + "="*50)
        print("【檢查 4：體重變化與平均名次】")
        valid_weight_df = df.dropna(subset=['weight_diff']).copy()
        # 將體重變化分組：減重 (<0), 持平 (0), 增重 (>0)
        valid_weight_df['weight_trend'] = np.where(valid_weight_df['weight_diff'] < 0, 'Decreased',
                                          np.where(valid_weight_df['weight_diff'] > 0, 'Increased', 'Same'))
        weight_result = valid_weight_df.groupby('weight_trend')['finish_pos'].agg(['mean', 'count'])
        print(weight_result)
        print("-> 預期：減重 的平均名次通常會較好（數字較小）")
        print("="*50)
        
        # ==========================================
        # 儲存為最終特徵檔，供 XGBoost 使用
        # ==========================================
        output_path = 'horse_form_features_ready.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ 所有特徵已計算完成並儲存為: {output_path}")
        print("下一步可以直接將此檔案送入 XGBoost 進行訓練！")
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}，請確認路徑是否正確！")
    except Exception as e:
        print(f"執行時發生錯誤：{e}")