"""混合检索（LCEL 版，第 30 / 50 / 52 / 53 课）。

向量检索（Milvus）+ BM25 关键词检索用 RunnableParallel 并行执行，
RRF 融合与 Rerank 精排用 RunnableLambda 包装，串成一条 LCEL 链。
"""
import json

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .bm25_retriever import bm25_search
from .rerank import rerank
from .config import CHUNKS_META_PATH
from .milvus_store import search as milvus_search
from .models import SearchResult


def load_meta(meta_path=CHUNKS_META_PATH):
    """加载 BM25 关键词检索用的文本语料（每次读盘，避免上传后缓存过期）。"""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def rrf_fuse(vector_results, bm25_results, top_k=3, k=60):
    """RRF 融合：按排名（而非原始分数）合并两路结果，异构分数也能公平比较。"""
    scores = {}
    by_idx = {}
    for r in vector_results + bm25_results:
        scores[r.idx] = scores.get(r.idx, 0) + 1.0 / (k + r.rank)
        by_idx[r.idx] = r   # 保留一路的 heading/content/source，用于结果组装

    sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for rank, idx in enumerate(sorted_indices, 1):
        src = by_idx[idx]
        results.append(SearchResult(
            rank=rank,
            score=scores[idx],
            heading=src.heading,
            content=src.content,
            idx=idx,
            source=src.source,
        ))
    return results


def _fuse_and_rerank(d, top_k=3):
    """第 52 课 RunnableLambda 包装：接收并行检索的 dict，做 RRF 融合 + Rerank 精排。"""
    fused = rrf_fuse(d["vector"], d["bm25"], top_k=len(d["vector"]))
    documents = [r.content for r in fused]
    rerank_results = rerank(d["question"], documents)
    results = []
    for i, r in enumerate(rerank_results[:top_k]):
        original = fused[r["index"]]
        results.append(SearchResult(
            rank=i + 1,
            score=r["relevance_score"],
            heading=original.heading,
            content=original.content,
            idx=original.idx,
            source=original.source,
        ))
    return results


def _build_retrieval_chain(top_k=3, candidate_num=10):
    """构建混合检索链（第 30 课 LCEL）。

    第 50 课 RunnableParallel：向量 + BM25 两路并行；
    第 53 课 RunnablePassthrough：把原始问题原样传给后面的 rerank 步骤。
    """
    return (
        RunnableParallel(
            vector=RunnableLambda(lambda q: milvus_search(q, top_k=candidate_num)),
            bm25=RunnableLambda(lambda q: bm25_search(q, load_meta(), top_k=candidate_num)),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(lambda d: _fuse_and_rerank(d, top_k=top_k))
    )


# 默认参数的检索链（供完整 RAG 链组合使用）
retrieval_chain = _build_retrieval_chain()


def hybrid_search_with_rerank(query, top_k=3, candidate_num=10):
    """混合检索入口：返回 SearchResult 列表（内部走 LCEL 检索链）。"""
    return _build_retrieval_chain(top_k=top_k, candidate_num=candidate_num).invoke(query)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python hybrid_retriever.py 你的问题")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    results = hybrid_search_with_rerank(query)
    print(f"问题：{query}\n")
    print(f"混合检索结果（Top{len(results)}）：\n")
    for r in results:
        print(f"【第{r.rank}名 | 相关性分数: {r.score:.4f} | 标题: {r.heading}】")
        print(r.content)
        print()
