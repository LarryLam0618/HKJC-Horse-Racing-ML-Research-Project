import pandas as pd
import numpy as np
import csv


LAG_COLS = [
    'final_split', 'fastest_split', 'avg_split', 'split_std',
    'pos_change', 'late_speed', 'late_speed_ratio', 'total_time'
]


def detect_delimiter(filepath: str, sample_size: int = 1024) -> str:
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        sample = f.read(sample_size)
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter


def load_csv_with_fallback(filepath: str) -> pd.DataFrame:
    try:
        delimiter = detect_delimiter(filepath)
        print(f"✅ 自動偵測到分隔符號: '{delimiter}'")
        return pd.read_csv(filepath, sep=delimiter, engine='python')
    except Exception as e:
        print(f"⚠️ 自動偵測失敗: {e}")

    delimiters = [',', '\t', '|', ';']
    for delim in delimiters:
        try:
            print(f"🔄 嘗試用分隔符號: '{delim}'")
            df = pd.read_csv(filepath, sep=delim, engine='python')
            if len(df.columns) > 3:
                print(f"✅ 成功用 '{delim}' 讀取！")
                return df
        except Exception:
            continue

    raise Exception("❌ 無法讀取 CSV 檔案，請檢查檔案格式")


def build_sectional_features(sectionals: pd.DataFrame) -> pd.DataFrame:
    sec = sectionals.copy()
    sec.columns = sec.columns.str.strip()

    required_cols = [
        'race_date', 'venue', 'race_no', 'horse_id',
        'section_index', 'split_200m_s', 'running_position', 'final_time_s'
    ]
    missing = [c for c in required_cols if c not in sec.columns]
    if missing:
        raise ValueError(f"❌ 缺少必要欄位: {missing}")

    sec['race_date'] = pd.to_datetime(sec['race_date'], errors='coerce')
    sec['race_no'] = pd.to_numeric(sec['race_no'], errors='coerce')
    sec['section_index'] = pd.to_numeric(sec['section_index'], errors='coerce')
    sec['split_200m_s'] = pd.to_numeric(sec['split_200m_s'], errors='coerce')
    sec['running_position'] = pd.to_numeric(sec['running_position'], errors='coerce')
    sec['final_time_s'] = pd.to_numeric(sec['final_time_s'], errors='coerce')

    sec = sec.dropna(subset=['race_date', 'venue', 'race_no', 'horse_id', 'section_index'])
    sec = sec.sort_values(['horse_id', 'race_date', 'race_no', 'section_index']).reset_index(drop=True)

    features = sec.groupby(['race_date', 'venue', 'race_no', 'horse_id']).agg(
        final_split=('split_200m_s', 'last'),
        fastest_split=('split_200m_s', 'min'),
        avg_split=('split_200m_s', 'mean'),
        split_std=('split_200m_s', 'std'),
        pos_change=('running_position', lambda x: x.iloc[0] - x.iloc[-1]),
        total_time=('final_time_s', 'first')
    ).reset_index()

    features['late_speed'] = features['avg_split'] - features['final_split']
    features['late_speed_ratio'] = np.where(
        features['avg_split'].notna() & (features['avg_split'] != 0),
        features['late_speed'] / features['avg_split'],
        np.nan
    )
    return features


def add_lagged_features(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df = df.sort_values(['horse_id', 'race_date', 'race_no']).reset_index(drop=True)

    for col in LAG_COLS:
        df[f'{col}_lag1'] = df.groupby('horse_id')[col].shift(1)

    df['sectional_history_count'] = df.groupby('horse_id').cumcount()
    df['has_sectional_history'] = (df['sectional_history_count'] > 0).astype(int)

    leak_check = (df['final_split'] == df['final_split_lag1']).fillna(False).mean()
    print(f"🔍 Leakage sanity check (final_split == lag1 比例): {leak_check:.4f}")

    keep_cols = [
        'race_date', 'venue', 'race_no', 'horse_id',
        'sectional_history_count', 'has_sectional_history'
    ] + [f'{c}_lag1' for c in LAG_COLS]

    return df[keep_cols].copy()


def check_time_coverage(sec_features: pd.DataFrame, full_results_path: str = 'results.csv'):
    print('\n' + '=' * 50)
    print('📅 sectional 資料時間範圍檢查')
    print('=' * 50)

    valid_dates = sec_features['race_date'].dropna()
    if len(valid_dates) == 0:
        print('⚠️ race_date 全部都是 NaT，無法檢查時間範圍！')
        return

    print('最早日期:', valid_dates.min())
    print('最晚日期:', valid_dates.max())

    print('\n按月份分佈:')
    monthly = sec_features.dropna(subset=['race_date']).copy()
    monthly['year_month'] = monthly['race_date'].dt.to_period('M')
    print(monthly['year_month'].value_counts().sort_index())

    print('\n' + '=' * 50)
    print('🐎 覆蓋率檢查 (每場平均馬數)')
    print('=' * 50)

    race_key = ['race_date', 'venue', 'race_no']
    n_races = sec_features.dropna(subset=race_key).groupby(race_key).ngroups
    n_rows = sec_features.dropna(subset=race_key).shape[0]
    avg_horses_per_race = n_rows / n_races if n_races > 0 else np.nan

    print(f'覆蓋賽事數: {n_races} 場')
    print(f'總馬匹記錄數: {n_rows} 行')
    print(f'平均每場馬數: {avg_horses_per_race:.2f}')

    if avg_horses_per_race < 8:
        print('⚠️ 警告：平均每場馬數偏低，可能只記錄部分馬匹（survivorship bias 風險）！')
    else:
        print('✅ 平均每場馬數正常，覆蓋率結構合理。')

    try:
        results3 = pd.read_csv(full_results_path)
        results3['race_date'] = pd.to_datetime(results3['race_date'], errors='coerce')
        n_races_full = results3.dropna(subset=['race_date', 'venue', 'race_no']).groupby(
            ['race_date', 'venue', 'race_no']
        ).ngroups
        coverage_pct = n_races / n_races_full * 100 if n_races_full > 0 else np.nan
        print(f'\n{full_results_path} 總賽事數: {n_races_full} 場')
        print(f'sectional 覆蓋率: {coverage_pct:.1f}%')
    except FileNotFoundError:
        print(f'\n⚠️ 找不到 {full_results_path}，跳過覆蓋率對比。')
    except Exception as e:
        print(f'\n⚠️ 對比覆蓋率時發生錯誤: {e}')


if __name__ == '__main__':
    print('📂 正在讀取 sectionals_clean.csv ...')
    try:
        sectionals = load_csv_with_fallback('sectionals_clean.csv')
        print(f'✅ 成功讀取 {len(sectionals)} 行，{len(sectionals.columns)} 個欄位')

        sec_features = build_sectional_features(sectionals)
        sec_lagged = add_lagged_features(sec_features)

        print('\n✅ 分段特徵構建完成！')
        print('\n📊 原始特徵前 10 行:')
        print(sec_features.head(10))
        print(f'\n原始 Shape: {sec_features.shape}')

        print('\n📊 Lag 後可訓練特徵前 10 行:')
        print(sec_lagged.head(10))
        print(f'\nLagged Shape: {sec_lagged.shape}')

        print('\n📊 原始分段特徵統計摘要:')
        print(sec_features[LAG_COLS].describe())

        print('\n📊 Lag 缺失率檢查:')
        lag_na = sec_lagged[[f'{c}_lag1' for c in LAG_COLS]].isna().mean().sort_values(ascending=False)
        print(lag_na)

        check_time_coverage(sec_features, full_results_path='results.csv')

        sec_features.to_csv('sectional_features_full.csv', index=False)
        sec_lagged.to_csv('sectional_features_lagged.csv', index=False)

        print('\n✅ 已儲存至: sectional_features_full.csv')
        print('✅ 已儲存至: sectional_features_lagged.csv')
        print('✅ 建議訓練模型時只使用 sectional_features_lagged.csv，避免 leakage')

    except FileNotFoundError:
        print("❌ 錯誤：搵唔到 'sectionals_clean.csv' 檔案！")
    except Exception as e:
        print(f'❌ 發生錯誤: {e}')
        import traceback
        traceback.print_exc()