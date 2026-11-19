"""Milvus 向量库封装（第 49 / 54 课）。

用 LangChain 的 Milvus 类替代原来的 FAISS：向量和元数据（标题 / 来源 / 下标）
存进同一个集合里。检索默认用 MMR（第 54 课）保证结果多样性，避免重复片段。
"""
import os

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

from .config import MILVUS_URI, MILVUS_COLLECTION
from .models import SearchResult

_embeddings = None
_vector_store = None


def get_embeddings():
    """返回 LangChain 嵌入模型，指向智谱 embedding-3（OpenAI 兼容接口）。"""
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
        try:
            _vector_store = Milvus(
                get_embeddings(),
                connection_args={"uri": MILVUS_URI},
                collection_name=MILVUS_COLLECTION,
                auto_id=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Milvus 连接失败（请确认已执行 docker start milvus-standalone）：{e}"
            ) from e
    return _vector_store


def add_documents(documents):
    """把 Document 列表写入 Milvus（第 49 课 add_documents）。"""
    try:
        return get_vector_store().add_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Milvus 写入失败：{e}") from e


def search(query, top_k=3, use_mmr=True):
    """向量相似度检索（第 54 课），默认用 MMR 保证多样性。"""
    try:
        vs = get_vector_store()
        if use_mmr:
            # MMR：既相关又多样，避免返回重复片段（fetch_k 多取候选再挑）
            docs = vs.max_marginal_relevance_search(
                query, k=top_k, fetch_k=top_k * 3, lambda_mult=0.6
            )
            results = []
            for rank, doc in enumerate(docs, 1):
                meta = doc.metadata
                results.append(SearchResult(
                    rank=rank,
                    score=0.0,   # MMR 不返回分数，占位（下游 RRF 只用 rank）
                    heading=meta.get("heading", "正文"),
                    content=doc.page_content,
                    idx=int(meta.get("idx", 0)),
                    source=meta.get("source", "未知来源"),
                ))
            return results

        # 普通带分搜索（similarity_search_with_score）
        scored = vs.similarity_search_with_score(query, k=top_k)
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
    except Exception as e:
        raise RuntimeError(f"Milvus 检索失败：{e}") from e
