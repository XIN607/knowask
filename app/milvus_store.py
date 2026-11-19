"""Milvus 向量库封装（第 49 / 54 / 55 课）。

用 LangChain 的 Milvus 类替代原来的 FAISS：向量和元数据（标题 / 来源 / 下标）
存进同一个集合里，不再需要「FAISS 索引文件 + JSON 元数据」两张皮对齐。
"""
import os

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

from .config import MILVUS_URI, MILVUS_COLLECTION
from .models import SearchResult

_embeddings = None
_vector_store = None


def get_embeddings():
    """返回 LangChain 嵌入模型，指向智谱 embedding-3（OpenAI 兼容接口）。

    和你在第 25/30 课学的 ChatOpenAI 是同一个套路：自定义 base_url + api_key，
    只是它管的是 embedding（文本→向量），不是对话。
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "embedding-3"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            api_key=os.getenv("EMBEDDING_API_KEY"),
        )
    return _embeddings


def get_vector_store():
    """返回（并缓存）Milvus 向量存储对象（第 49 课）。

    Collection 不存在时会自动创建；已存在则直接连接，不会清空已有数据。
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = Milvus(
            get_embeddings(),
            connection_args={"uri": MILVUS_URI},
            collection_name=MILVUS_COLLECTION,
            auto_id=True,
        )
    return _vector_store


def add_documents(documents):
    """把 Document 列表写入 Milvus（第 49 课 add_documents）。"""
    return get_vector_store().add_documents(documents)


def search(query, top_k=3):
    """向量相似度检索（第 54 课 similarity_search_with_score），返回 SearchResult 列表。"""
    scored = get_vector_store().similarity_search_with_score(query, k=top_k)
    results = []
    for rank, (doc, score) in enumerate(scored, 1):
        meta = doc.metadata
        results.append(SearchResult(
            rank=rank,
            score=float(score),
            heading=meta.get("heading", "正文"),
            content=doc.page_content,
            idx=int(meta.get("idx", 0)),
            source=meta.get("source", "未知来源"),
        ))
    return results
