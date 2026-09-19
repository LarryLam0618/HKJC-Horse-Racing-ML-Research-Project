import pandas as pd
import numpy as np
import csv

def build_weight_features(results: pd.DataFrame) -> pd.DataFrame:
    """
    從 results.csv 構建體重相關特徵
    所有特徵都使用 shift(1)，確保只用到過去比賽的數據
    """
    df = results.copy()
    
    # 確保格式正確同埋係數值
    df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
    df['declared_weight'] = pd.to_numeric(df['declared_weight'], errors='coerce')
    
    # 加入 race_no 確保排序穩定
    df = df.sort_values(['horse_id', 'race_date', 'race_no']).reset_index(drop=True)
    
    # --- 特徵 1: 體重變化（絕對值） ---
    df['prev_weight'] = df.groupby('horse_id')['declared_weight'].shift(1)
    df['weight_delta'] = df['declared_weight'] - df['prev_weight']
    
    # --- 特徵 2: 體重變化（百分比）---
    # 加入除零保護
    df['weight_delta_pct'] = np.where(
        df['prev_weight'].notna() & (df['prev_weight'] != 0),
        df['weight_delta'] / df['prev_weight'],
        np.nan
    )
    
    # --- 特徵 3: 體重趨勢 ---
    df['weight_ma3'] = df.groupby('horse_id')['declared_weight'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['weight_trend'] = df['declared_weight'] - df['weight_ma3']
    
    # --- 特徵 4: 同場體重偏離 ---
    df['race_avg_weight'] = df.groupby(
        ['race_date', 'venue', 'race_no']
    )['declared_weight'].transform('mean')
    df['weight_vs_field'] = df['declared_weight'] - df['race_avg_weight']
    
    # --- 特徵 5: 體重波動度 ---
    df['weight_volatility'] = df.groupby('horse_id')['declared_weight'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    )
    
    # 清理過渡欄位
    df = df.drop(columns=['prev_weight', 'weight_ma3', 'race_avg_weight'])
    
    return df

# ... (保留你寫嘅 detect_delimiter, load_csv_with_fallback 同埋 if __name__ == '__main__': 邏輯)
def detect_delimiter(filepath: str, sample_size: int = 1024) -> str:
    """
    自動偵測 CSV 檔案嘅分隔符號
    """
    with open(filepath, 'r') as f:
        sample = f.read(sample_size)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        return delimiter


def load_csv_with_fallback(filepath: str) -> pd.DataFrame:
    """
    嘗試多種方式讀取 CSV，自動處理分隔符號問題
    """
    # 嘗試 1: 自動偵測
    try:
        delimiter = detect_delimiter(filepath)
        print(f"✅ 自動偵測到分隔符號: '{delimiter}'")
        return pd.read_csv(filepath, sep=delimiter, engine='python')
    except Exception as e:
        print(f"⚠️ 自動偵測失敗: {e}")
    
    # 嘗試 2: 常見分隔符號列表
    delimiters = [',', '\t', '|', ';', ' ']
    for delim in delimiters:
        try:
            print(f"🔄 嘗試用分隔符號: '{delim}'")
            df = pd.read_csv(filepath, sep=delim, engine='python')
            # 如果成功讀取而且有足夠欄位，就返回
            if len(df.columns) > 3:
                print(f"✅ 成功用 '{delim}' 讀取！")
                return df
        except Exception:
            continue
    
    # 嘗試 3: 用 regex 處理多字符分隔符
    try:
        print("🔄 嘗試用 regex 分隔符號: '\\s+\\|\\s+'")
        df = pd.read_csv(filepath, sep='\s+\|\s+', engine='python')
        if len(df.columns) > 3:
            print("✅ 成功用 regex 讀取！")
            return df
    except Exception as e:
        print(f"⚠️ Regex 方法失敗: {e}")
    
    # 全部失敗
    raise Exception("❌ 無法讀取 CSV 檔案，請檢查檔案格式")


# === 使用方式 ===
if __name__ == '__main__':
    # 自動偵測並讀取 CSV
    try:
        print("📂 正在讀取 results.csv ...")
        results = load_csv_with_fallback('results.csv')
        
        print(f"✅ 成功讀取 {len(results)} 行，{len(results.columns)} 個欄位")
        print(f"📋 欄位名稱: {list(results.columns)[:10]} ...")
        
        # 檢查必要欄位是否存在
        required_cols = ['horse_id', 'race_date', 'declared_weight', 'venue', 'race_no']
        missing_cols = [col for col in required_cols if col not in results.columns]
        
        if missing_cols:
            print(f"⚠️ 缺少必要欄位: {missing_cols}")
            print("📋 請檢查 CSV 嘅欄位名稱是否同以下匹配:")
            print("   - horse_id")
            print("   - race_date")
            print("   - declared_weight")
            print("   - venue")
            print("   - race_no")
        else:
            # 構建體重特徵
            results = build_weight_features(results)
            
            # 顯示結果
            print("\n✅ 體重特徵構建完成！")
            print("\n📊 前 10 行結果預覽:")
            print(results[['horse_id', 'race_date', 'declared_weight', 
                         'weight_delta', 'weight_delta_pct', 'weight_trend',
                         'weight_vs_field', 'weight_volatility']].head(10))
            
            # 顯示統計摘要
            print("\n📊 體重特徵統計摘要:")
            weight_features = ['weight_delta', 'weight_delta_pct', 'weight_trend', 
                             'weight_vs_field', 'weight_volatility']
            print(results[weight_features].describe())
            
            # 儲存結果
            output_file = 'results_with_weight_features.csv'
            results.to_csv(output_file, index=False)
            print(f"\n✅ 已儲存至: {output_file}")
            
    except FileNotFoundError:
        print("❌ 錯誤：搵唔到 'results.csv' 檔案！")
        print("請確保:")
        print("  1. 檔案名稱係 'results.csv'")
        print("  2. 同呢個 script 喺同一個目錄")
        print("  3. 或者用完整路徑，例如: '/path/to/your/results.csv'")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()