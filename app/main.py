from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from fastapi.responses import FileResponse
import sqlite3
from datetime import datetime

from .config import UPLOAD_DIR, INDEX_HTML, CHAT_HISTORY_DB


app = FastAPI(title="知问 KnowAsk API", version="1.0")

os.makedirs(UPLOAD_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(CHAT_HISTORY_DB)
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
    return FileResponse(INDEX_HTML)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 1. 保存上传的文件到 uploads/ 目录
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 2. 调用入库流程：解析 → 切分 → 向量化 → 建索引
    from .vector_store import build_index
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
    from .hybrid_retriever import hybrid_search_with_rerank
    from .qa_chain import check, stream_answer, REFUSE_TEXT

    # 1. 检索
    contexts = hybrid_search_with_rerank(request.question, top_k=3)

    # 2. 拒答判断（结构化 found，非流式）
    result = check(request.question, contexts)

    # 3. 存用户消息
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    c = conn.cursor()
    c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("user", request.question))
    conn.commit()
    conn.close()

    # 4. SSE 返回：found=False 直接拒答；found=True 流式回答
    def generate():
        full_answer = ""
        if not result.found:
            full_answer = REFUSE_TEXT
            yield f"data: {full_answer}\n\n"
        else:
            for chunk in stream_answer(request.question, contexts):
                full_answer += chunk
                yield f"data: {chunk}\n\n"

        # 流式结束后存 AI 回答
        conn = sqlite3.connect(CHAT_HISTORY_DB)
        c = conn.cursor()
        c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("ai", full_answer))
        conn.commit()
        conn.close()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/history")
async def get_history():
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    c = conn.cursor()
    c.execute("SELECT role, content, created_at FROM messages ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "time": r[2]} for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
