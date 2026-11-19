"""大模型客户端（第 25/30 课 ChatOpenAI，OpenAI 兼容接口接 DeepSeek）。"""
import os

from langchain_openai import ChatOpenAI

from . import config  # 触发 config 的 load_dotenv，确保环境变量已加载

_llm = None


def get_llm():
    """返回（并缓存）ChatOpenAI 实例，指向 DeepSeek（OpenAI 兼容接口）。

    和原来用 openai 库直连 DeepSeek 是同一套，只是换成了 LangChain 的封装，
    这样后面才能用 | 管道把它接进 LCEL 链。
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=0.7,
        )
    return _llm
