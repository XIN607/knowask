import os
from dotenv import load_dotenv
from openai import OpenAI

# 1.加载 .env 文件里的配置到环境变量
load_dotenv()

# 2.从环境变量读取配置
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL","https://api.deepseek.com/v1")
)

# 3. 发起对话请求：stream=True 表示流式输出（一个字一个字吐出来）
response = client.chat.completions.create(
    model = os.getenv("LLM_MODEL","deepseek-chat"),
    messages=[
        {"role": "user","content": "你好，请用一句话介绍你自己。"}
    ],
    stream = True
)

# 4. 流式打印  逐段接收模型的回复并打印
print("模型回复：",end="",flush=True)
for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content,end="",flush=True)
print()  #最后换行
