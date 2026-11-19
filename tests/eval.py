import sys
from pathlib import Path

# 让本脚本无论从哪个目录运行都能找到项目根目录下的 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.hybrid_retriever import hybrid_search_with_rerank
from app.qa_chain import check


# 检索测试集：每条是 (问题, 期望命中的关键词列表)
retrieval_cases = [
    ("什么是RAG", ["检索增强", "Retrieval-Augmented"]),
    ("我叫什么名字", ["小明", "软件工程"]),
    ("我在做什么项目", ["RAG", "知识库"]),
]

# 拒答（防幻觉）测试集：资料里没有这些问题，found 应为 False
refusal_cases = [
    "今天天气怎么样",
    "明天股票会涨吗",
]


def evaluate_retrieval():
    """评测检索：关键词是否出现在 Top3 的某个片段里。"""
    total = len(retrieval_cases)
    hit = 0
    for query, keywords in retrieval_cases:
        results = hybrid_search_with_rerank(query, top_k=3)
        hit_flag = any(kw in r.content for r in results for kw in keywords)
        if hit_flag:
            hit += 1
            print(f"√ '{query}' → 命中")
        else:
            print(f"× '{query}' → 未命中")
    print(f"\n检索命中率: {hit}/{total} = {hit/total*100:.0f}%")


def evaluate_refusal():
    """评测拒答（防幻觉）：无关问题的 found 应为 False。"""
    total = len(refusal_cases)
    hit = 0
    for query in refusal_cases:
        contexts = hybrid_search_with_rerank(query, top_k=3)
        result = check(query, contexts)
        if not result.found:
            hit += 1
            print(f"√ [拒答] '{query}' → found=False（正确拒答）")
        else:
            print(f"× [拒答] '{query}' → found=True（不该回答）")
    print(f"\n拒答准确率: {hit}/{total} = {hit/total*100:.0f}%")


if __name__ == "__main__":
    evaluate_retrieval()
    print()
    evaluate_refusal()
