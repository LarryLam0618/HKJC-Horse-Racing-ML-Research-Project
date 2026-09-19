import pandas as pd
import numpy as np

def parse_final_position(running_pos_raw):
    """
    解析 running_position_raw 字串
    例如 "5 4 1" → 最終名次 1
    """
    try:
        parts = str(running_pos_raw).strip().split()
        return int(parts[-1])
    except:
        return np.nan


def build_trial_features(trials: pd.DataFrame) -> pd.DataFrame:
    """
    從 barrier_trials_clean.csv 構建試閘特徵
    
    策略：對每匹馬，按 trial_date 排序，
    然後用 asof join（取比賽日期之前最近一次試閘）
    
    Input:  trials DataFrame
    Output: 每匹馬每次試閘的特徵，之後用 asof join 合併到 results
    """
    tr = trials.copy()
    tr['trial_date'] = pd.to_datetime(tr['trial_date'])
    
    # 解析最終名次
    tr['trial_final_pos'] = tr['running_position_raw'].apply(
        parse_final_position
    )
    
    # 計算試閘馬數（分母）
    tr['trial_field_size'] = tr.groupby(
        ['trial_date', 'venue', 'batch']
    )['horse_id'].transform('count')
    
    # 試閘名次比率（0=最好, 1=最差）
    tr['trial_pos_ratio'] = tr['trial_final_pos'] / tr['trial_field_size']
    
    # 按馬匹+日期排序
    tr = tr.sort_values(['horse_id', 'trial_date'])
    
    # 每匹馬的試閘歷史特徵
    tr_features = tr[[
        'horse_id', 'trial_date', 'trial_final_pos',
        'trial_pos_ratio', 'time_s', 'lbw_raw'
    ]].copy()
    
    tr_features = tr_features.rename(columns={
        'trial_final_pos': 'last_trial_pos',
        'trial_pos_ratio': 'last_trial_pos_ratio',
        'time_s': 'last_trial_time',
        'lbw_raw': 'last_trial_lbw',
        'trial_date': 'trial_date'
    })
    
    return tr_features


def merge_trial_features_to_results(
    results: pd.DataFrame, 
    trial_features: pd.DataFrame
) -> pd.DataFrame:
    """
    用 asof join 將最近一次試閘特徵合併到 results
    確保只用比賽日期之前的試閘數據
    """
    df = results.copy()
    tf = trial_features.copy()
    
    df['race_date'] = pd.to_datetime(df['race_date'])
    tf['trial_date'] = pd.to_datetime(tf['trial_date'])
    
    # 確保兩邊都按 horse_id + 日期排序
    df = df.sort_values(['horse_id', 'race_date'])
    tf = tf.sort_values(['horse_id', '在深圳'])
    
    # asof join: 取每匹馬在比賽日期之前最近嘅一次試閘
    merged = pd.merge_asof(
        df,
        tf,
        left_on='race_date',
        right_on='trial_date',
        by='horse_id',
        direction='backward'  # 只向後睇（過去）
    )
    
    # 計算試閘距今天數
    merged['days_since_trial'] = (
        merged['race_date'] - merged['trial_date']
    ).dt.days
    
    return merged


# === 使用方式 ===
if __name__ == '__main__':
    trials = pd.read_csv('data/barrier_trials_clean.csv', sep=' | ')
    tf = build_trial_features(trials)
    print(tf.head(10))
    print(f"Unique horses: {tf['horse_id'].nunique()}")