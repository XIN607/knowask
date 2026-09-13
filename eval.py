from hybrid_retriever import hybrid_search_with_rerank

# 测试集：每条是 (问题, 期望命中的关键词列表)
# 关键词只要出现在检索到的片段里就算命中
test_cases = [
    ("什么是RAG", ["检索增强", "Retrieval-Augmented"]),
    ("我叫什么名字", ["小明", "软件工程"]),
    ("我在做什么项目", ["RAG", "知识库"]),
    ("今天天气怎么样", [""]),    # 不期望命中任何片段（防幻觉测试）
]

def evaluate():
    total = len(test_cases)
    hit = 0

    for query, keywords in test_cases:
        results = hybrid_search_with_rerank(query, top_k=3)

        # 如果关键词列表是空的，说明期望"检索不到相关内容"
        if not keywords:
            # 检查 Top1 的相关性分数是否低于 0.99（不相关）
            if results[0]["score"] < 0.99:
                hit += 1
                print(f"√ [防幻觉] '{query}' → Top1分数={results[0]['score']:.4f}（正确，没瞎答）")
            else:
                print(f"× [防幻觉] '{query}' → Top1分数={results[0]['score']:.4f}（有问题，不该命中）")
            continue

        # 正常测试：检查关键词是否出现在 Top3 的某个片段里
        hit_flag = False
        for r in results:
            for kw in keywords:
                if kw in r["content"]:
                    hit_flag = True
                    break

        if hit_flag:
            hit += 1
            print(f"√ '{query}' → 命中")
        else:
            print(f"× '{query}' → 未命中")
            print(f"   实际检索到：{results[0]['heading']} - {results[0]['content'][:50]}...")

    print(f"\n命中率: {hit}/{total} = {hit/total*100:.0f}%")

if __name__ == "__main__":
    evaluate()
