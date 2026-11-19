"""向量检索（独立使用）：调用 Milvus 做相似度搜索。"""
from .milvus_store import search as milvus_search


def search(query, top_k=3):
    """输入一个问题，返回最相关的 top_k 个片段（SearchResult 列表）。"""
    return milvus_search(query, top_k=top_k)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python retriever.py 你的问题")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = search(query)

    print(f"问题: {query}\n")
    print(f"检索到 {len(results)} 个相关片段: \n")
    for r in results:
        print(f"[第{r.rank}名 | 分数: {r.score:.4f} | 标题: {r.heading}]")
        print(r.content)
        print()
