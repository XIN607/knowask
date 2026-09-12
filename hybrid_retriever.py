import json 
import faiss
import numpy as np
from retriever import get_embedding
from bm25_retriever import bm25_search
from rerank import rerank

_index = None
_meta = None

def load_index_and_meta(index_path="faiss_index.bin",meta_path="chunks_meta.json"):
    global _index,_meta
    if _index is None:
        _index = faiss.read_index(index_path)
        with open(meta_path,"r",encoding="utf-8") as f:
            _meta = json.load(f)
    return _index,_meta

def vector_search(query,top_k=3):
    index,meta = load_index_and_meta()
    query_vec = get_embedding(query)
    distances,indices = index.search(np.array([query_vec]).astype("float32"),top_k)
    results = []
    for i,idx in enumerate(indices[0]):
        if idx < 0:
            continue
        results.append({
            "rank":i+1,
            "score":float(distances[0][i]),
            "heading":meta[idx]["heading"],
            "content":meta[idx]["content"],
            "idx":idx
        })
    return results

def rrf_fuse(vector_results,bm25_results,top_k=3,k=60):
    scores = {}
    for r in vector_results:
        idx = r["idx"]
        scores[idx] = scores.get(idx,0) + 1.0/(k+r["rank"])
    for r in bm25_results:
        idx = r["idx"]
        scores[idx] = scores.get(idx,0) + 1.0/(k+r["rank"])

    sorted_indices = sorted(scores.keys(),key=lambda i:scores[i],reverse = True)[:top_k]
    _,meta =load_index_and_meta()
    results = []
    for rank,idx in enumerate(sorted_indices,1):
        results.append({
            "rank":rank,
            "score":scores[idx],
            "heading":meta[idx]["heading"],
            "content":meta[idx]["content"],
            "idx":idx
        })
    return results

def hybrid_search(query, top_k=3):
    _, meta = load_index_and_meta()
    vector_results = vector_search(query, top_k=top_k)
    bm25_results = bm25_search(query, meta, top_k=top_k)
    fused_results = rrf_fuse(vector_results, bm25_results, top_k=top_k)
    return fused_results

def hybrid_search_with_rerank(query,top_k=3,candidate_num=10):
    _,meta = load_index_and_meta()
    # 1.粗排：向量 + BM25 +RRF,取 candidate_num 个候选
    vector_results = vector_search(query,top_k=candidate_num)
    bm25_results = bm25_search(query,meta,top_k=candidate_num)
    fused_results = rrf_fuse(vector_results,bm25_results,top_k=candidate_num)
    # 2.精排：Rerank 重新排序
    documents=[r["content"] for r in fused_results]
    rerank_results = rerank(query,documents)
    # 3.取 top_k 个作为最终结果
    results=[]
    for i,r in enumerate(rerank_results[:top_k]):
        original = fused_results[r["index"]]
        results.append({
            "rank":i+1,
            "score":r["relevance_score"],
            "heading":original["heading"],
            "content":original["content"],
            "idx":original["idx"]
        })
    return results


if __name__=="__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python hybrid_retriever.py 你的问题")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    results = hybrid_search_with_rerank(query)
    print(f"问题：{query}\n")
    print(f"混合检索结果（Top{len(results)}）：\n")
    for r in results:
        print(f"【第{r['rank']}名 | 相关性分数: {r['score']:.4f} | 标题: {r['heading']}】")
        print(r["content"])
        print()
