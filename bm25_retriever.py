import jieba
from rank_bm25 import BM25Okapi

def tokenize(text):
    """中文分词：把一段文字切成词语列表"""
    return list(jieba.cut(text))

def build_bm25_index(chunks):
    """
    用所有文档片段建立 BMM25 索引
    chunks: 片段列表，每个元素是字典，包含 “content字段”
    返回: BM25 索引对象
    """
    tokenized_corpus = [tokenize(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def bm25_search(query,chunks,top_k=3):
    """
    BM25 关键词检索
    query: 用户问题
    chunks: 片段列表
    top_k: 返回前几个
    返回：按 BM25 分数排序的片段列表（带排名和分数）
    """

    bm25 = build_bm25_index(chunks)
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # 按分数从高到底排序 取 top_k
    ranked_indices = sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:top_k]

    results = []
    for rank,idx in enumerate(ranked_indices,1):
        results.append({
            "rank":rank,
            "score":float(scores[idx]),
            "heading":chunks[idx]["heading"],
            "content":chunks[idx]["content"],
            "idx":idx
        })
    return results

if __name__=="__main__":
    import json
    with open("chunks_meta.json","r",encoding="utf-8") as f:
        chunks = json.load(f)

    query = "RAG是什么"
    results = bm25_search(query,chunks)
    print(f"问题：{query}\n")
    for r in results:
        print(f"[第{r['rank']}名 | BM25分数: {r['score']:.4f} | 标题：{r['heading']}]")
        print(r['content'])
        print()