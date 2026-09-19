import pandas as pd
import numpy as np

print("正在讀取原始資料...")
# 讀取你的原始 horse_form
form_df = pd.read_csv('horse_form.csv')
form_df.columns = form_df.columns.str.strip()

# 讀取含有真實日期的 results-3.csv (請確認檔名與路徑正確)
try:
    results_df = pd.read_csv('results.csv')
    results_df.columns = results_df.columns.str.strip()
except FileNotFoundError:
    print("錯誤：找不到 results-3.csv，請確認檔案名稱與路徑！")
    exit()

# 統一欄位名稱以利比對 (假設 results-3.csv 裡的日期欄位叫 run_date 或 race_date)
if 'run_date' not in results_df.columns and 'race_date' in results_df.columns:
    results_df.rename(columns={'race_date': 'run_date'}, inplace=True)

# 確保比對欄位型態一致 (避免 01 與 1 的問題)
for col in ['horse_id', 'draw', 'win_odds', 'finish_pos']:
    if col in form_df.columns and col in results_df.columns:
        form_df[col] = form_df[col].astype(str)
        results_df[col] = results_df[col].astype(str)

print("正在進行複合鍵 Fuzzy Match 還原日期...")
# 使用 horse_id + draw + win_odds + finish_pos 作為複合鍵進行合併
keys = ['horse_id', 'draw', 'win_odds', 'finish_pos']
matched_results = results_df[keys + ['run_date']].drop_duplicates(subset=keys)

# 將還原的日期合併回 form_df
form_final = form_df.merge(matched_results, on=keys, how='left', suffixes=('', '_restored'))

# 如果原始檔有 run_date 但為空，用還原的 run_date 補上
if 'run_date_restored' in form_final.columns:
    form_final['run_date'] = form_final['run_date'].fillna(form_final['run_date_restored'])
    form_final = form_final.drop(columns=['run_date_restored'])

missing_count = form_final['run_date'].isna().sum()
total_count = len(form_final)
print(f"\n還原完成！剩餘缺失日期: {missing_count} / {total_count} ({missing_count/total_count:.1%})")

# 儲存還原日期後的完整檔案
output_path = 'horse_form_final.csv'
form_final.to_csv(output_path, index=False)
print(f"已將還原日期後的資料儲存為: {output_path}")