# 知问 KnowAsk - 基于 RAG 的智能知识库问答系统

一个基于 RAG 的知识库问答系统（个人全栈实践项目），支持文档上传、混合检索、Rerank 精排、多轮对话与流式输出。

## 功能特性

- 📄 **多格式文档解析**：支持 PDF / Word / Markdown / TXT 文档上传与自动解析
- ✂️ **智能文档切分**：标题感知切分 + 滑动窗口重叠，保证语义完整
- 🔍 **混合检索**：向量检索（Milvus 语义匹配）+ BM25（关键词匹配）+ RRF 融合
- 🎯 **Rerank 精排**：使用 Rerank 模型对粗排结果重新排序，提升检索准确率
- 💬 **多轮对话**：会话级历史记忆，支持追问与指代
- ⚡ **流式对话**：SSE 流式输出，打字机效果
- 🛡️ **防幻觉**：结构化 found 判断，资料中没有的内容确定性拒答
- 📚 **引用溯源**：问答输出参考来源（标题路径 + 相关性分数）
- 🌐 **Web 界面**：前后端分离，RESTful API + 简洁聊天页面

## 技术栈

| 层级 | 技术 | 说明 |
|---|---|---|
| 大语言模型 | DeepSeek | 对话生成（流式输出） |
| 向量化模型 | 智谱 embedding-3 | 文本转向量（2048 维） |
| 重排序模型 | 智谱 rerank | 候选结果精排 |
| 向量数据库 | Milvus | 向量 + 元数据统一存储 |
| 关键词检索 | BM25 + jieba | 中文分词 + 关键词匹配 |
| 应用框架 | LangChain | LCEL 链、PydanticOutputParser、Runnable 组合 |
| 后端框架 | FastAPI | 高性能 Web 框架，自动生成 API 文档 |
| 服务器 | Uvicorn | ASGI 服务器，支持异步 |
| 前端 | HTML/CSS/JavaScript | 原生前端，SSE 流式接收 |
| 文档处理 | pypdf / python-docx | PDF / Word 解析 |

## 系统架构

```
┌─────────────────────────────────────────┐
│           前端 (index.html)              │
│  上传文档 + 聊天界面 + SSE 流式渲染       │
└──────────────┬──────────────────────────┘
               │ HTTP / SSE
┌──────────────▼──────────────────────────┐
│           后端 (FastAPI)                 │
│  GET /health  POST /upload  POST /chat   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         RAG 核心引擎（LCEL）             │
│                                         │
│  入库：解析 → 切分 → 向量化 → Milvus      │
│                                         │
│  问答：问题 → 并行检索(向量+BM25)          │
│              → RRF 融合 → Rerank 精排    │
│              → 拒答判断(found)           │
│              → LLM 流式生成              │
│              → 回答 + 引用来源           │
└─────────────────────────────────────────┘
```

## 项目结构

```text
knowask/
├── app/                        # 后端源码
│   ├── config.py               # 路径与 .env 集中管理
│   ├── models.py               # Pydantic 数据模型（Chunk / SearchResult / CheckResult）
│   ├── llm.py                  # ChatOpenAI 客户端（DeepSeek）
│   ├── milvus_store.py         # Milvus 向量库封装 + 嵌入缓存
│   ├── main.py                 # FastAPI 入口（上传 / 问答 / 历史接口）
│   ├── parse_doc.py            # PDF / Word / Markdown 解析
│   ├── chunker.py              # 标题感知 + 滑动窗口切分
│   ├── vector_store.py         # 文档入库（Milvus + BM25 语料）
│   ├── retriever.py            # 向量检索
│   ├── bm25_retriever.py       # BM25 关键词检索
│   ├── hybrid_retriever.py     # 双路检索 + RRF + Rerank（LCEL）
│   ├── rerank.py               # Rerank 精排
│   ├── qa_chain.py             # 拒答判断 + 流式回答链
│   └── rag_qa.py               # CLI 问答入口
├── scripts/                    # 独立演示脚本
├── tests/                      # 评测脚本（eval.py）与测试文档
├── static/index.html           # 前端页面（SSE 流式渲染）
├── data/                       # 运行时生成：BM25 语料 / 对话历史
├── uploads/                    # 上传的文档
└── .env                        # 环境变量
```

> 说明：向量和元数据存在 Milvus（外部服务，默认 `localhost:19530`）；`data/` 只保留
> BM25 关键词检索语料和 SQLite 对话历史。`data/`、`uploads/` 均不入库（已在 `.gitignore` 忽略）。

## 快速开始

### 环境要求

- Python 3.10+
- Milvus（Docker，默认 `localhost:19530`）
- DeepSeek API Key
- 智谱 AI API Key

### 安装步骤

```bash
# 1. 启动 Milvus（如未启动）
docker start milvus-standalone

# 2. 克隆项目
git clone <your-repo-url>
cd knowask

# 3. 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate  # Windows

# 4. 安装依赖
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 5. 配置环境变量
在项目根目录创建 .env 文件，填入 API Key（代码读取的环境变量：
LLM_API_KEY、LLM_BASE_URL、LLM_MODEL、EMBEDDING_API_KEY、EMBEDDING_BASE_URL、EMBEDDING_MODEL）

# 6. 启动服务
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

浏览器打开 http://127.0.0.1:8000（前端页面），或 http://127.0.0.1:8000/docs（接口文档）。
