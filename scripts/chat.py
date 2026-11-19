import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL","https://api.deepseek.com/v1")
)
model = os.getenv("LLM_MODEL","deepseek-chat")

# 对话历史列表：第一句是“系统指令”，设定 AI 的角色
messages = [
    {"role":"system","content":"你是一个乐于助人的AI助手,回答要简要清晰"}
]

print("开始对话吧！输入 exit 或 quit 退出。")

while True:
    # 1.等用户输入
    user_input = input("你：")

    # 2.输入 exit/quit 就退出
    if user_input.lower() in("exit","input"):
        print("再见！")
        break

    # 3.把用户问题加入历史
    messages.append({"role":"user","content":user_input})

    # 4.把完整历史发给模型
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    # 5.流式接收回答，同时拼接完整回复
    print("AI: ",end="",flush=True)
    reply=""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content,end="",flush=True)
            reply += chunk.choices[0].delta.content
    print()

    # 6. 把模型回答也加入历史
    messages.append({"role":"assistant","content":reply})
