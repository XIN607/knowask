from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from fastapi.responses import FileResponse
import sqlite3
from datetime import datetime

from .config import UPLOAD_DIR, INDEX_HTML, CHAT_HISTORY_DB


app = FastAPI(title="知问 KnowAsk API", version="1.0")

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def init_db():
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            session_id TEXT DEFAULT 'default',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 旧库迁移：补 session_id 列（已存在则跳过）
    try:
        c.execute("ALTER TABLE messages ADD COLUMN session_id TEXT DEFAULT 'default'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def save_message(session_id, role, content):
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (role, content, session_id) VALUES (?, ?, ?)",
        (role, content, session_id),
    )
    conn.commit()
    conn.close()


def get_recent_history(session_id, limit=6):
    """取该会话最近 N 条消息，按时间正序返回 [(role, content), ...]。"""
    conn = sqlite3.connect(CHAT_HISTORY_DB)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))


init_db()


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "知问服务运行中"}

@app.get("/")
async def read_root():
    return FileResponse(INDEX_HTML)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 1. 校验文件类型
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    # 2. 保存上传的文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 3. 入库（解析 → 切分 → 向量化 → Milvus）
    try:
        from .vector_store import build_index
        build_index(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档入库失败：{e}")

    return {"filename": file.filename, "message": "文档入库成功"}


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.post("/chat")
async def chat(request: QuestionRequest):
    from .hybrid_retriever import hybrid_search_with_rerank
    from .qa_chain import check, stream_answer, REFUSE_TEXT

    session_id = request.session_id or "default"
    history = get_recent_history(session_id)   # 多轮上下文（不含当前问题）

    # 检索 + 拒答判断（可能因 Milvus / API 失败而异常）
    try:
        contexts = hybrid_search_with_rerank(request.question, top_k=3)
        result = check(request.question, contexts, history=history)
    except Exception as e:
        error_text = f"服务暂时不可用：{e}"
        save_message(session_id, "user", request.question)
        save_message(session_id, "ai", error_text)

        def error_gen():
            yield f"data: {error_text}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    save_message(session_id, "user", request.question)

    def generate():
        full_answer = ""
        try:
            if not result.found:
                full_answer = REFUSE_TEXT
                yield f"data: {full_answer}\n\n"
            else:
                for chunk in stream_answer(request.question, contexts, history=history):
                    full_answer += chunk
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            full_answer = f"服务暂时不可用：{e}"
            yield f"data: {full_answer}\n\n"
        save_message(session_id, "ai", full_answer)

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
