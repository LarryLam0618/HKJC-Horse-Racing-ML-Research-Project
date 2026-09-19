import os
import json
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# =====================================================================
# 1. 載入環境變數
# =====================================================================
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("⚠️ 警告：未找到 OPENAI_API_KEY，請檢查 .env 檔案")
    print("💡 如果冇 .env 檔案，請建立一個並加入：")
    print("   OPENAI_API_KEY=你的API金鑰")
    print("   OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4/")
    exit(1)

# =====================================================================
# 2. 設定 LLM（用免費版 GLM-4-Flash）
# =====================================================================
glm_free_llm = "openai/glm-4-flash"

print("🤖 使用模型: GLM-4-Flash (免費)")

# =====================================================================
# 3. 自訂 CSV 讀取工具
# =====================================================================
@tool("讀取CSV檔案")
def read_csv_file(file_path: str) -> str:
    """
    讀取 CSV 檔案並轉換成 Agent 容易理解嘅格式。
    
    參數:
    - file_path: CSV 檔案嘅路徑
    
    回傳:
    - JSON 格式嘅數據摘要
    """
    try:
        if not os.path.exists(file_path):
            all_csv = []
            for root, dirs, files in os.walk("."):
                if "__pycache__" in root or ".git" in root:
                    continue
                for f in files:
                    if f.endswith(".csv"):
                        rel_path = os.path.relpath(os.path.join(root, f), ".")
                        all_csv.append(rel_path)
            
            if all_csv:
                csv_list = "\n".join([f"  📄 {f}" for f in all_csv[:20]])
                return f"""❌ 搵唔到檔案 '{file_path}'。

📂 你嘅專案入面有呢啲 CSV 檔案（頭20個）：
{csv_list}

💡 請用正確嘅檔案名再試一次。"""
            else:
                return "❌ 搵唔到任何 CSV 檔案"
        
        print(f"📂 正在讀取: {file_path}")
        df = pd.read_csv(file_path)
        
        total_rows = len(df)
        display_rows = min(30, total_rows)
        
        result = {
            "檔案名稱": os.path.basename(file_path),
            "檔案路徑": file_path,
            "總記錄數": total_rows,
            "欄位數量": len(df.columns),
            "欄位名稱": list(df.columns),
            "數據樣本（頭{}行）".format(display_rows): df.head(display_rows).to_dict(orient='records'),
            "基本統計": {}
        }
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                result["基本統計"][col] = {
                    "平均值": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "最大值": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "最小值": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                }
        
        result["備註"] = f"✅ 成功讀取 {total_rows} 行數據，顯示頭 {display_rows} 行"
        
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        
    except Exception as e:
        return f"❌ 讀取 CSV 失敗: {str(e)}"

# =====================================================================
# 4. 搵出最新賽事日期嘅工具（新增！）
# =====================================================================
@tool("搵出最近賽事日期")
def find_latest_race_date() -> str:
    """
    讀取 results_clean.csv 並搵出最近嘅賽事日期同該日所有賽事。
    會自動搵出數據入面最新嘅日期（2026年）。
    """
    try:
        # 嘗試讀取 results_clean.csv
        file_path = "results_clean.csv"
        if not os.path.exists(file_path):
            return "❌ 搵唔到 results_clean.csv，請確認檔案存在。"
        
        df = pd.read_csv(file_path)
        
        # 搵日期欄位
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            return "❌ 搵唔到日期欄位"
        
        # 轉做日期格式
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 搵最新日期
        latest_date = df[date_col].max()
        latest_date_str = latest_date.strftime('%Y-%m-%d')
        
        # 搵該日所有賽事
        latest_races = df[df[date_col] == latest_date]
        
        # 統計該日賽事
        race_nos = latest_races['race_no'].unique().tolist()
        horses = latest_races['horse_name'].tolist()[:20]  # 最多20匹
        
        result = {
            "最新賽事日期": latest_date_str,
            "該日總記錄數": len(latest_races),
            "賽事場次": race_nos,
            "參賽馬匹（頭20匹）": horses,
            "建議": f"請用 read_csv_file 讀取 results_clean.csv，然後過濾 date='{latest_date_str}' 嘅數據"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        
    except Exception as e:
        return f"❌ 錯誤: {str(e)}"

# =====================================================================
# 5. 模擬數據工具（備用）
# =====================================================================
@tool("讀取賽馬特徵與預測數據工具")
def fetch_hkjc_model_data() -> str:
    """備用工具：如果 CSV 讀取失敗，用呢個模擬數據。"""
    mock_data = {
        "賽事資訊": "沙田草地 1200米 - A賽道",
        "場地預測": "好地 (Good to Firm)",
        "數據來源": "⚠️ 模擬數據（備用）",
        "模型預測馬匹": [
            {"馬名": "金鎗六十", "檔位": 5, "LightGBM預測勝率": "34.5%", "當前獨贏賠率": "2.1", "騎師": "何澤堯"},
            {"馬名": "浪漫勇士", "檔位": 11, "LightGBM預測勝率": "28.2%", "當前獨贏賠率": "3.5", "騎師": "麥道朗"},
            {"馬名": "美好世界", "檔位": 2, "LightGBM預測勝率": "12.1%", "當前獨贏賠率": "9.5", "騎師": "潘頓"},
            {"馬名": "加州星球", "檔位": 8, "LightGBM預測勝率": "10.8%", "當前獨贏賠率": "12.0", "騎師": "艾道拿"}
        ]
    }
    return json.dumps(mock_data, ensure_ascii=False, indent=2)

# =====================================================================
# 6. 建立 5 個專業 Agent
# =====================================================================

# Agent 1: 數據工程師
data_engineer = Agent(
    role="HKJC 賽馬數據特徵工程師",
    goal="""
    負責讀取多個 CSV 數據檔案，並搵出最新嘅賽事數據。
    
    重要指示：
    1. 首先用 find_latest_race_date 工具搵出數據中最新嘅日期
    2. 然後用 read_csv_file 工具讀取 'results_clean.csv'
    3. 過濾出最新日期嘅賽事數據
    4. 將數據整理成結構化格式
    """,
    backstory="""
    你是數據專家，精通處理大量賽馬數據。
    你最擅長從數據中搵出最新、最 relevant 嘅資訊。
    你知道 2026 年嘅數據先係最新嘅！
    """,
    llm=glm_free_llm,
    tools=[read_csv_file, find_latest_race_date, fetch_hkjc_model_data],
    max_iter=5,
    verbose=True
)

# Agent 2: 騎師分析師
jockey_analyst = Agent(
    role="騎師表現與狀態分析師",
    goal="""
    分析騎師嘅歷史表現、勝率、黏地適應能力。
    
    分析重點：
    1. 騎師嘅整體勝率
    2. 騎師喺不同場地（好地/黏地）嘅表現
    3. 騎師同特定馬匹嘅合作歷史
    4. 騎師近期狀態
    """,
    backstory="""
    你係香港賽馬會最資深嘅騎師分析師，對每個騎師嘅風格、強弱項瞭如指掌。
    你曾經準確預測過多次冷門騎師爆冷勝出。
    """,
    llm=glm_free_llm,
    max_iter=5,
    verbose=True
)

# Agent 3: 場地專家
track_specialist = Agent(
    role="賽道與場地狀況專家",
    goal="""
    評估場地狀況對賽事嘅影響。
    
    分析重點：
    1. 好地 vs 黏地對不同檔位嘅影響
    2. 場地變化對馬匹表現嘅影響
    3. 歷史數據中類似場地狀況嘅賽果
    4. 天氣預報對賽事嘅潛在影響
    """,
    backstory="""
    你係賽道管理專家，擁有20年場地分析經驗。
    你精通不同場地狀況對賽馬表現嘅影響，尤其擅長預測場地變化帶來嘅賽果變數。
    """,
    llm=glm_free_llm,
    max_iter=5,
    verbose=True
)

# Agent 4: 賠率價值專家
value_analyst = Agent(
    role="賽馬賠率與價值投資專家",
    goal="""
    對比模型預測勝率與市場賠率，找出價值投注機會。
    
    分析步驟：
    1. 計算每匹馬嘅期望值 (EV)
    2. 對比模型勝率 vs 市場隱含勝率
    3. 結合騎師同場地分析，調整 EV
    4. 找出最佳價值投注
    """,
    backstory="你係統計學家，精通博弈論同期望值計算。",
    llm=glm_free_llm,
    max_iter=5,
    verbose=True
)

# Agent 5: 首席決策官
chief_bot = Agent(
    role="首席馬賽風險決策官",
    goal="""
    綜合所有專家分析，輸出完整繁體中文賽事報告。
    
    報告內容：
    1. 賽事基本資訊（必須係最新日期！）
    2. 場地分析（好地 vs 黏地）
    3. 騎師分析
    4. 價值投注分析
    5. 推薦注碼策略
    6. 資金配置
    7. 風險提示
    """,
    backstory="""
    你係團隊領導者，重視風險管理同全局分析。
    你只會分析最新嘅賽事數據，唔會用 2008 年嘅舊數據！
    """,
    llm=glm_free_llm,
    max_iter=3,
    verbose=True
)

# =====================================================================
# 7. 設計 Tasks
# =====================================================================

task_1_fetch = Task(
    description="""
    請執行以下步驟讀取最新嘅賽馬數據：

    第 1 步：搵出最新日期
    - 用 find_latest_race_date 工具
    - 呢個工具會話你知數據中最新嘅日期（應該係 2026 年）

    第 2 步：讀取 CSV 數據
    - 用 read_csv_file 讀取 'results_clean.csv'
    - 留意數據入面嘅 'date' 欄位

    第 3 步：過濾最新日期嘅數據
    - 只保留最新日期嘅記錄
    - 列出該日所有賽事嘅馬匹資料

    第 4 步：整理數據
    - 列出每匹馬嘅：馬名、檔位、賠率、騎師、體重
    - 用 Markdown 表格格式

    ⚠️ 重要：必須用最新日期（2026年），唔好用 2008 年！
    """,
    expected_output="**2026年最近一場賽事**嘅馬匹數據表（Markdown 格式）",
    agent=data_engineer
)

task_2_jockey = Task(
    description="""
    根據數據工程師提供嘅最新賽事數據，分析騎師表現：

    1. 列出所有參賽騎師
    2. 分析每個騎師嘅歷史勝率（從數據中估算）
    3. 評估騎師喺黏地嘅適應能力
    4. 評估騎師同馬匹嘅合作經驗
    5. 提供騎師評分（1-10分）
    6. 指出邊個騎師最有優勢
    """,
    expected_output="騎師分析報告（包含評分同建議）",
    agent=jockey_analyst
)

task_3_track = Task(
    description="""
    根據最新賽事數據，分析場地影響：

    1. 分析檔位對賽果嘅影響（歷史數據）
    2. 假設場地由「好地」轉「黏地」：
       - 評估對內檔（1-4檔）嘅影響
       - 評估對中檔（5-8檔）嘅影響
       - 評估對外檔（9-14檔）嘅影響
    3. 評估場地變化對每匹馬嘅影響
    4. 調整每匹馬嘅勝率預測
    """,
    expected_output="場地分析報告（包含調整後勝率）",
    agent=track_specialist
)

task_4_value = Task(
    description="""
    綜合數據工程師、騎師分析師、場地專家嘅報告：

    1. 計算每匹馬嘅調整後期望值 (EV)：
       EV = (調整後勝率 × 賠率) - 1

    2. 對比調整後勝率 vs 市場隱含勝率

    3. 找出最佳價值投注機會：
       - 列出所有 EV 為正數嘅馬匹
       - 按 EV 高低排序
       - 推薦最佳 3 個投注選擇

    4. 指出應該避免嘅馬匹（EV 為負數）
    """,
    expected_output="價值投資分析報告（含 EV 排名）",
    agent=value_analyst
)

task_5_final = Task(
    description="""
    綜合所有專家分析，輸出最終報告：

    報告結構：
    
    # 🏆 香港賽馬會賽事黃金指南
    
    ## 📋 賽事基本資訊
    - 日期（**必須係 2026 年最新日期！**）
    - 場地、路程、參賽馬匹數量
    
    ## 🌤️ 天氣與場地分析
    - 當前場地狀況
    - 場地變化預測（好地→黏地）
    - 檔位影響評估
    
    ## 🏇 騎師分析
    - 各騎師評分
    - 騎師優勢分析
    
    ## 📊 價值投注分析
    - 期望值 (EV) 排名表
    - 最佳價值投注推薦
    
    ## 💰 推薦注碼策略
    - 獨贏投注建議
    - 位置投注建議
    - 資金配置（%）
    
    ## ⚠️ 風險提示
    - 主要風險因素
    - 注意事項
    
    ## 📈 總結
    - 整體建議

    ⚠️ 日期必須係 2026 年，唔可以係 2008 年！
    """,
    expected_output="完整繁體中文賽事黃金指南（2026年最新數據）",
    agent=chief_bot
)

# =====================================================================
# 8. 組建 Crew
# =====================================================================
racing_crew = Crew(
    agents=[data_engineer, jockey_analyst, track_specialist, value_analyst, chief_bot],
    tasks=[task_1_fetch, task_2_jockey, task_3_track, task_4_value, task_5_final],
    process=Process.sequential,
    verbose=True
)

# =====================================================================
# 9. 輔助函數：儲存報告
# =====================================================================
def save_report(result, filename: str = None):
    """儲存報告去檔案"""
    if hasattr(result, 'raw'):
        content = result.raw
    elif hasattr(result, 'output'):
        content = str(result.output)
    else:
        content = str(result)
    
    os.makedirs("reports", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 儲存 TXT
    txt_file = f"reports/racing_report_{timestamp}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ TXT 報告已儲存: {txt_file}")
    
    # 2. 儲存 HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港賽馬會賽事黃金指南</title>
    <style>
        body {{
            font-family: 'PingFang HK', 'Microsoft YaHei', sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #c0392b; border-bottom: 3px solid #c0392b; padding-bottom: 15px; text-align: center; }}
        h2 {{ color: #2980b9; margin-top: 30px; border-left: 4px solid #2980b9; padding-left: 15px; }}
        h3 {{ color: #27ae60; margin-top: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f9f9f9; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .warning {{ background: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin: 15px 0; }}
        .success {{ background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 香港賽馬會賽事黃金指南</h1>
        <p style="text-align:center; color:#666;">生成日期: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
        <p style="text-align:center; color:#666;">🤖 由 CrewAI + 智譜 GLM-4-Flash 分析生成</p>
        <hr>
        <div class="content">
            {content.replace('\n', '<br>')}
        </div>
        <div class="footer">
            <p>⚠️ 本報告僅供參考，投注涉及風險，請理性投資</p>
            <p>📧 如有疑問，請諮詢專業顧問</p>
        </div>
    </div>
</body>
</html>"""
    
    html_file = f"reports/racing_report_{timestamp}.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ HTML 報告已儲存: {html_file}")
    
    return txt_file, html_file

# =====================================================================
# 10. 主程式
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 香港賽馬 AI 智囊團 (5 個專業 Agent)")
    print("=" * 70)
    print("📊 使用模型: GLM-4-Flash (完全免費 🆓)")
    print("👥 Agent 團隊:")
    print("   1. 數據工程師 - 讀取多個 CSV + 搵最新日期")
    print("   2. 騎師分析師 - 騎師表現評估")
    print("   3. 場地專家 - 場地狀況分析")
    print("   4. 賠率專家 - 價值投注計算")
    print("   5. 決策官 - 最終報告輸出")
    print("📅 將會自動搵出 2026 年最新賽事")
    print("=" * 70)
    print()
    
    try:
        result = asyncio.run(racing_crew.kickoff_async())
        
        print("\n" + "=" * 70)
        print("🏆 最終智譜大模型決策報告")
        print("=" * 70)
        
        if hasattr(result, 'raw'):
            print(result.raw)
        else:
            print(result)
        
        print("=" * 70)
        
        save_report(result)
        
        print("\n🎉 分析完成！")
        print("📁 報告已儲存至 reports/ 資料夾")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 使用者中斷執行")
        
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()