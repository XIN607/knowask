import os
import requests
from dotenv import load_dotenv

load_dotenv()

# 1. 检查 Key
key = os.getenv("EMBEDDING_API_KEY")
print(f"Key是否存在: {bool(key)}")
if key:
    print(f"Key前10位: {key[:10]}")
    print(f"Key长度: {len(key)}")

# 2. 调用 rerank API，打印完整响应
url = "https://open.bigmodel.cn/api/paas/v4/rerank"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "rerank",
    "query": "什么是RAG",
    "documents": ["RAG是检索增强生成", "我叫小明"]
}

response = requests.post(url, headers=headers, json=payload)
print(f"\n状态码: {response.status_code}")
print(f"响应体: {response.text}")
