#!/bin/bash
# 香港賽馬 AI 自動分析腳本

# 進入專案目錄
cd "$(dirname "$0")/../.." || exit 1   # repo root

# 設定日誌目錄
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 日誌檔案
LOG_FILE="$LOG_DIR/racing_$(date +\%Y\%m\%d).log"

# 開始記錄
echo "========================================" >> "$LOG_FILE"
echo "🚀 開始分析: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 執行 Python 程式（修正版）
python3 research/experiments/hkjc_crew_system.py >> "$LOG_FILE" 2>&1

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo "✅ 分析成功完成: $(date)" >> "$LOG_FILE"
else
    echo "❌ 分析失敗: $(date)" >> "$LOG_FILE"
fi

echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"