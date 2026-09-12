import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 智谱 Embedding 客户端（和 vector_store.py 里一样）
embedding_client = OpenAI(
    api_key = os.getenv("EMBEDDING_API_KEY"),
    base_url= os.getenv("EMBEDDING_BASE_URL","https://open.bigmodel.cn/api/paas/v4")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","embedding-3")

def get_embedding(text):
    """把一段文字转成向量"""
    response=embedding_client.embeddings.create(
        model= EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def search(query,index_path="faiss_index.bin",meta_path="chunks_meta.json",top_k=3):
    """
    请输入一个问题，返回最相关的 top_k 个文档片段
    """
    # 1.加载之前保存的 FAISS 索引和元数据
    index = faiss.read_index(index_path)
    with open(meta_path,"r",encoding="utf-8") as f:
        meta = json.load(f)

    # 2.把用户问题向量化
    query_vec = get_embedding(query)

    # 3.FAISS搜索：找距离最近的 top_k 个向量
    #   distances = 每个结果的距离(越小越相似)
    #   indices = 每个结果在索引里的编号(从0开始)
    distances,indices = index.search(
        np.array([query_vec]).astype("float32"),
        top_k
    )

    # 4.根据编号去元数据里查原文,组装成结果
    results = []
    for i,idx in enumerate(indices[0]):
        if idx < 0:           # FAISS 有时会返回 -1 表示空结果,跳过
            continue
        results.append({
            "rank": i+1,
            "score": float(distances[0][i]),
            "heading": meta[idx]["heading"],
            "content": meta[idx]["content"]
        })  

    return results  

if __name__ =="__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python retriever.py 你的问题")
        sys.exit(1)

    # 把命令行参数拼起来(问题可能有空格,比如 "什么是 RAG")
    query = " ".join(sys.argv[1:])
    results = search(query)

    print(f"问题:{query}\n")
    print(f"检索到 {len(results)} 个相关片段: \n")
    for r in results:
        print(f"[第{r['rank']}名 | 距离: {r['score']:.4f} | 标题 :{r['heading']}]")
        print(r["content"])
        print()