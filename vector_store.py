import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from parse_doc import parse_document
from chunker import chunk_document

load_dotenv()

# 智谱 Embedding 客户端（复用 openai 库，因为智谱 API 是 OpenAI 兼容的）
embedding_client = OpenAI(
    api_key = os.getenv("EMBEDDING_API_KEY"),
    base_url = os.getenv("EMBEDDING_BASE_URL","https://open.bigmodel.cn/api/paas/v4")
)
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL","embedding-3")

def get_embedding(text):
    """调用 Embedding API,把一段文本转成向量(一串数字)"""
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input = text
    )
    return response.data[0].embedding

def build_index(doc_path,index_path="faiss_index.bin",meta_path="chunks_meta.json"):
    """
    端到端流水线：解析文档->切分-向量化-建立 FAISS 索引-保存到本地
    """
    # 1.解析文档 + 切分（复用前两步写好的模块）
    text = parse_document(doc_path)
    chunks = chunk_document(text)
    print(f"文档切分为{len(chunks)}个片段")

    # 2.逐个向量化
    print("正在向量化...")
    embeddings=[]
    for i,chunk in enumerate(chunks):
        vec = get_embedding(chunk.content)
        embeddings.append(vec)
        print(f"  片段{i+1}/{len(chunks)}向量化完成（维度{len(vec)}")

    # 3.建立 FAISS 索引(L2 距离 = 欧氏距离，距离越小越相似)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    print(f"FAISS 索引建立完成，共{index.ntotal}条向量，维度{dimension}")

    # 4.保存索引和元数据到本地
    faiss.write_index(index,index_path)
    meta = [
        {"content":c.content,"heading":c.heading,"char_count":c.char_count}
        for c in chunks
    ]
    with open(meta_path,"w",encoding="utf-8")as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)
    print(f"索引已保存到 {index_path}")
    print(f"元数据已保存到 {meta_path}")

    return index,meta

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("用法: python vector_store.py 文件路劲")
        sys.exit(1)
    build_index(sys.argv[1])