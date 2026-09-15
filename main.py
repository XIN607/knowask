from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.responses import FileResponse
import sqlite3
from datetime import datetime


app = FastAPI(title="知问 KnowAsk API", version="1.0")
UPLOAD_DIR = "uploads"
load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "知问服务运行中"}
@app.get("/")
async def read_root():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 1. 保存上传的文件到 uploads/ 目录
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 2. 调用入库流程：解析 → 切分 → 向量化 → 建索引
    from vector_store import build_index
    build_index(file_path)

    # 3. 返回结果
    return {
        "filename": file.filename,
        "message": "文档入库成功"
    }

class QuestionRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: QuestionRequest):
    from hybrid_retriever import hybrid_search_with_rerank

    contexts = hybrid_search_with_rerank(request.question, top_k=3)

    prompt = "你是一个知识库问答助手。请只根据下面的资料回答用户问题。如果资料里没有相关信息，请回答'根据现有资料无法回答这个问题'。\n\n"
    for i, ctx in enumerate(contexts, 1):
        prompt += f"【资料{i}】{ctx['heading']}：{ctx['content']}\n\n"
    prompt += f"用户问题：{request.question}\n\n回答："

    # 先把用户问题存进数据库
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("user", request.question))
    conn.commit()
    conn.close()

    full_answer = ""  # 用来收集完整回答

    def generate():
        nonlocal full_answer
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_answer += content  # 拼接完整回答
                yield f"data: {content}\n\n"

        # 流式输出完后，把 AI 回答存进数据库
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("ai", full_answer))
        conn.commit()
        conn.close()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/history")
async def get_history():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("SELECT role, content, created_at FROM messages ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "time": r[2]} for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
