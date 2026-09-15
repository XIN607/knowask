# 知问 KnowAsk - 基于 RAG 的智能知识库问答系统

一个基于 RAG 的知识库问答系统（个人全栈实践项目），支持文档上传、混合检索、Rerank 精排、流式对话。

## 功能特性

- 📄 **多格式文档解析**：支持 PDF / Word / Markdown / TXT 文档上传与自动解析
- ✂️ **智能文档切分**：标题感知切分 + 滑动窗口重叠，保证语义完整
- 🔍 **混合检索**：向量检索（语义匹配）+ BM25（关键词匹配）+ RRF 融合
- 🎯 **Rerank 精排**：使用 Rerank 模型对粗排结果重新排序，提升检索准确率
- 💬 **流式对话**：SSE 流式输出，打字机效果，体验媲美 ChatGPT
- 🛡️ **防幻觉**：Prompt 约束，资料中没有的内容 AI 不会编造
- 📚 **引用溯源**：命令行问答输出参考来源（标题路径 + 相关性分数），可追溯原文片段
- 🌐 **Web 界面**：前后端分离，提供 RESTful API + 简洁聊天页面

## 技术栈

| 层级 | 技术 | 说明 |
|---|---|---|
| 大语言模型 | DeepSeek | 对话生成（流式输出） |
| 向量化模型 | 智谱 embedding-3 | 文本转向量（2048维） |
| 重排序模型 | 智谱 rerank | 候选结果精排 |
| 向量数据库 | FAISS | 高性能向量索引与检索 |
| 关键词检索 | BM25 + jieba | 中文分词 + 关键词匹配 |
| 后端框架 | FastAPI | 高性能 Web 框架，自动生成 API 文档 |
| 服务器 | Uvicorn | ASGI 服务器，支持异步 |
| 前端 | HTML/CSS/JavaScript | 原生前端，SSE 流式接收 |
| 文档处理 | pypdf / python-docx | PDF / Word 解析 |

## 系统架构
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
│           RAG 核心引擎                   │
│                                         │
│  入库：解析 → 切分 → 向量化 → FAISS 索引  │
│                                         │
│  问答：问题 → 粗排 (向量 + BM25+RRF)       │
│              → 精排 (Rerank)             │
│              → Prompt 组装               │
│              → LLM 流式生成              │
│              → 回答 + 引用来源           │
└─────────────────────────────────────────┘


## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key
- 智谱 AI API Key

### 安装步骤

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd knowask

# 2. 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate  # Windows

# 3. 安装依赖
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. 配置环境变量
在项目根目录创建 .env 文件，填入 API Key（代码读取的环境变量：LLM_API_KEY、LLM_BASE_URL、LLM_MODEL、EMBEDDING_API_KEY、EMBEDDING_BASE_URL、EMBEDDING_MODEL）

# 5. 启动服务
.\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000

