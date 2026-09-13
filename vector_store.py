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
    支持多文档追加：如果已有索引，新文档的片段会追加进去
    """
    # 1.解析文档 + 切分（复用前两步写好的模块）
    text = parse_document(doc_path)
    chunks = chunk_document(text)
    print(f"文档切分为{len(chunks)}个片段")

    # 2.加载已有索引（如果存在），没有就新建
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"加载已有索引，当前 {len(meta)} 个片段")
    except:
        dim = len(get_embedding("测试"))
        index = faiss.IndexFlatL2(dim)
        meta = []
        print("新建索引")

    # 3.逐个向量化并追加到索引
    print("正在向量化...")
    source_name = os.path.basename(doc_path)  # 只取文件名，不要完整路径
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk.content)
        index.add(np.array([vec]).astype("float32"))
        # 元数据里记录来源文件名
        meta.append({
            "idx": len(meta),
            "content": chunk.content,
            "heading": chunk.heading,
            "char_count": chunk.char_count,
            "source": source_name
        })
        print(f"  片段{i+1}/{len(chunks)}向量化完成")

    # 4.保存索引和元数据到本地
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"索引已更新，共 {len(meta)} 个片段")
    return index, meta

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("用法: python vector_store.py 文件路径")
        sys.exit(1)
    build_index(sys.argv[1])
