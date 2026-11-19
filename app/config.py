"""全局配置：集中管理项目路径与环境变量加载。

运行时生成的文件（向量索引、元数据、对话历史、上传文档）统一放在项目根目录下的
data/、uploads/ 目录中，路径都从这里派生。这样无论从哪个目录启动服务，
都不会因为相对路径找不到文件而报错。
"""
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：本文件在 app/ 下，上一级即项目根
BASE_DIR = Path(__file__).resolve().parent.parent

# 从项目根目录加载 .env（用绝对路径定位，不依赖启动时的当前目录）
load_dotenv(BASE_DIR / ".env")

# 目录
DATA_DIR = BASE_DIR / "data"        # 向量索引 / 元数据 / 对话历史
UPLOAD_DIR = BASE_DIR / "uploads"   # 用户上传的文档
STATIC_DIR = BASE_DIR / "static"    # 前端静态页面

# 运行时数据文件（统一用字符串路径，兼容 faiss / sqlite3 等库）
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
CHUNKS_META_PATH = str(DATA_DIR / "chunks_meta.json")
CHAT_HISTORY_DB = str(DATA_DIR / "chat_history.db")
INDEX_HTML = str(STATIC_DIR / "index.html")

# 确保目录存在（幂等，重复执行无副作用）
for _dir in (DATA_DIR, UPLOAD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
