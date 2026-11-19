"""RAG 链路评测（pytest）。

注意：这些是集成测试，依赖 Milvus 已启动、API Key 已配置。
运行：python -m pytest tests/ -v
"""
import pytest

from app.hybrid_retriever import hybrid_search_with_rerank
from app.qa_chain import check


RETRIEVAL_CASES = [
    ("什么是RAG", ["检索增强", "Retrieval-Augmented"]),
    ("我叫什么名字", ["小明", "软件工程"]),
    ("我在做什么项目", ["RAG", "知识库"]),
]

REFUSAL_CASES = [
    "今天天气怎么样",
    "明天股票会涨吗",
]


@pytest.mark.parametrize("query,keywords", RETRIEVAL_CASES)
def test_retrieval_hits(query, keywords):
    """检索：关键词应出现在 Top3 的某个片段里。"""
    results = hybrid_search_with_rerank(query, top_k=3)
    contents = " ".join(r.content for r in results)
    assert any(kw in contents for kw in keywords), f"'{query}' 未命中关键词 {keywords}"


@pytest.mark.parametrize("query", REFUSAL_CASES)
def test_refusal(query):
    """拒答：无关问题的 found 应为 False。"""
    contexts = hybrid_search_with_rerank(query, top_k=3)
    result = check(query, contexts)
    assert result.found is False, f"'{query}' 应拒答，但 found=True"


def test_multi_turn_pronoun():
    """多轮：带历史能正确理解指代（「他」→ 小明）。"""
    contexts = hybrid_search_with_rerank("他是什么专业的", top_k=3)
    history = [("user", "我叫什么名字"), ("assistant", "你叫小明")]
    result = check("他是什么专业的", contexts, history=history)
    assert result.found is True, "多轮指代理解失败"
