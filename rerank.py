import os
import requests
from dotenv import load_dotenv

load_dotenv()


def rerank(query, documents, model="rerank", top_n=None):
    url = "https://open.bigmodel.cn/api/paas/v4/rerank"
    api_key = os.getenv("EMBEDDING_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "query": query,
        "documents": documents
    }
    if top_n:
        payload["top_n"] = top_n

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["results"]


if __name__ == "__main__":
    query = "什么是RAG"
    documents = [
        "RAG 全称是 Retrieval-Augmented Generation，即检索增强生成。",
        "我叫小明，是一名软件工程专业的学生。",
        "我正在搭建一个基于 RAG 的知识库问答系统。"
    ]
    results = rerank(query, documents)
    print(f"问题：{query}\n")
    for r in results:
        print(f"索引: {r['index']} | 相关性分数: {r['relevance_score']:.4f}")
        print(f"内容: {documents[r['index']]}\n")
