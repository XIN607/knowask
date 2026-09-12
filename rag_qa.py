import os 
from dotenv import load_dotenv
from openai import OpenAI
from hybrid_retriever import hybrid_search_with_rerank

load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL","https://api.deepseek.com/v1")
)
LLM_MODEL = os.getenv("LLM_MODEL","deepseek-chat")

# 把检索到的片段和用户问题组装成一个完整的 Prompt (提示词),发给大模型
def build_prompt(query,contexts):
    context_str  = "\n\n".join([f"[片段{i+1}] {c['content']}" for i,c in enumerate(contexts)])
    prompt = f"""你是一个知识库问答助手.请根据下面的参考资料回答用户的问题.
    
    要求:
    1.只根据参考资料回答,不要编造参考资料里没有的信息
    2.如果参考资料里没有答案,直接说"根据现有资料无法回答这个问题"
    3.回答要简洁清晰

    参考资料:
    {context_str}

    用户问题:{query}

    请回答:
    """

    return prompt

# 完整 RAG 流程
def rag_answer(query,top_k=3):
    # 1.检索
    contexts = hybrid_search_with_rerank(query, top_k=top_k)
    # 2.组装 Prompt
    prompt = build_prompt(query,contexts)
    # 3.调用大模型
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"user","content":prompt}],
        stream = True
    )
    # 4.流式打印h回答
    print("AI回答: ",end="",flush=True)
    answer = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content,end="",flush=True)
            answer += chunk.choices[0].delta.content
    print("\n")
    # 5.打印引用来源
    print("参考来源")
    for c in contexts:
        print(f" [{c['rank']}] [{c['heading']}] (距离: {c['score']:.4f})")

    return answer,contexts

if __name__ =="__main__":
    import sys
    if len(sys.argv)<2:
        print("用法 python rag_qa.py 你的问题")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print(f"问题: {query}\n")
    rag_answer(query)