"""拒答判断 + 流式回答（第 28 / 29 / 30 / 31 / 38 课）。

两段式：先用 PydanticOutputParser 让 LLM 结构化输出 found（是否命中），
命中后再用流式回答链逐段输出正文，既保留打字机效果，又实现确定性拒答。
"""
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .models import CheckResult
from .llm import get_llm

# 固定拒答文案（found=False 时返回）
REFUSE_TEXT = "根据现有资料无法回答这个问题"


def _format_contexts(contexts):
    """把 SearchResult 列表拼成一段参考资料文本。"""
    return "\n\n".join(
        f"[片段{i + 1}] {c.heading}：{c.content}" for i, c in enumerate(contexts)
    )


# 第 38 课：拒答判断链（结构化，非流式）—— 只输出 found + confidence
_check_parser = PydanticOutputParser(pydantic_object=CheckResult)
_check_prompt = ChatPromptTemplate.from_template("""
你是一个知识库问答助手。请判断：下面的资料里是否有能回答用户问题的内容？

资料：
{context}

用户问题：{question}

{format_instructions}
""").partial(format_instructions=_check_parser.get_format_instructions())

# 第 30 课 LCEL：prompt | model | parser
check_chain = _check_prompt | get_llm() | _check_parser

# 第 28 课 StrOutputParser + 第 31 课流式：回答链（纯文本，逐段输出）
_answer_prompt = ChatPromptTemplate.from_template("""
你是一个知识库问答助手，请根据下面的资料简洁准确地回答用户问题。

资料：
{context}

用户问题：{question}

回答：
""")
answer_chain = _answer_prompt | get_llm() | StrOutputParser()


def check(question, contexts):
    """判断资料里是否有答案，返回 CheckResult（含 found 布尔值）。"""
    return check_chain.invoke({"question": question, "context": _format_contexts(contexts)})


def stream_answer(question, contexts):
    """流式生成回答，返回生成器（逐段 yield 文本）。"""
    return answer_chain.stream({"question": question, "context": _format_contexts(contexts)})
