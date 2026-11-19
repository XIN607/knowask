"""CLI 问答入口：检索 + 拒答判断 + 流式回答。"""
from .hybrid_retriever import hybrid_search_with_rerank
from .qa_chain import check, stream_answer, REFUSE_TEXT


def rag_answer(query, top_k=3):
    # 1. 检索
    contexts = hybrid_search_with_rerank(query, top_k=top_k)
    # 2. 拒答判断（结构化 found）
    result = check(query, contexts)
    # 3. 输出（found=False 返回固定拒答文案，否则流式打印回答）
    if not result.found:
        print(f"AI回答: {REFUSE_TEXT}")
    else:
        print("AI回答: ", end="", flush=True)
        for chunk in stream_answer(query, contexts):
            print(chunk, end="", flush=True)
        print()
    print(f"(found={result.found}, 置信度={result.confidence:.2f})")
    # 4. 打印引用来源
    print("参考来源")
    for c in contexts:
        print(f" [{c.rank}] [{c.heading}] (相关性: {c.score:.4f})")
    return result, contexts


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python rag_qa.py 你的问题")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print(f"问题: {query}\n")
    rag_answer(query)
