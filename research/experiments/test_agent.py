import json
import os

import requests

# 1. API key comes from the environment — never hardcode it:  export ZHIPU_API_KEY=...
MY_ZHIPU_KEY = os.environ["ZHIPU_API_KEY"]

# 2. 精準的智譜清言 v4 接口網址
url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

headers = {
    "Authorization": f"Bearer {MY_ZHIPU_KEY}",
    "Content-Type": "application/json"
}

# 3. 發送測試資料 (使用您額度最大的 glm-4-air)
data = {
    "model": "glm-4-air",
    "messages": [
        {
            "role": "user",
            "content": "請用繁體中文對我說一聲：『哈囉！VS Code 連線成功！』"
        }
    ],
    "max_tokens": 50,
    "temperature": 0.7
}

print("🚀 正在執行物理安全測試，直連智譜 AI 伺服器...")
print(f"📡 當前使用的發送網址: {url}")

try:
    # 4. 發送標準 POST 請求
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=15)
    
    # 5. 解析回傳結果
    if response.status_code == 200:
        result_json = response.json()
        reply_content = result_json['choices'][0]['message']['content']
        print("\n--- 🏆 智能體回覆 ---")
        print(reply_content)
        print("--------------------")
    else:
        print(f"\n❌ 連線失敗！狀態碼: {response.status_code}")
        print("💡 提示：如果狀態碼依然是 405，說明您的 API Key 填寫錯誤、已過期，或當前網路被防火牆攔截。")
        print("伺服器返回內容:", response.text)

except Exception as e:
    print(f"\n❌ 發生異常錯誤: {e}")
