"""文档入库：解析 → 切分 → 写入 Milvus（向量 + 元数据）→ 更新 BM25 语料。"""
import os
import json

from langchain_core.documents import Document

from .parse_doc import parse_document
from .chunker import chunk_document
from .config import CHUNKS_META_PATH
from .milvus_store import add_documents


def build_index(doc_path, meta_path=CHUNKS_META_PATH):
    """端到端入库流程：解析 → 切分 → 写入 Milvus + 更新 BM25 语料。

    支持多文档追加：新文档的片段会继续累加下标（idx），不会覆盖旧数据。
    """
    # 1. 解析文档 + 切分
    text = parse_document(doc_path)
    chunks = chunk_document(text)
    print(f"文档切分为 {len(chunks)} 个片段")

    # 2. 加载已有 BM25 语料（同时用来确定新片段的下标起点）
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        meta = []
    start_idx = len(meta)

    # 3. 构造 Document（第 49 课的数据单元）+ 追加到 BM25 语料
    #    page_content 会被向量化；metadata 不会被向量化，但会作为字段一起存进 Milvus
    source_name = os.path.basename(doc_path)
    documents = []
    for i, chunk in enumerate(chunks):
        idx = start_idx + i
        meta.append({
            "idx": idx,
            "content": chunk.content,
            "heading": chunk.heading,
            "char_count": chunk.char_count,
            "source": source_name,
        })
        documents.append(Document(
            page_content=chunk.content,
            metadata={"heading": chunk.heading, "source": source_name, "idx": idx},
        ))

    # 4. 更新 BM25 语料（JSON，供关键词检索用）
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 5. 写入 Milvus（向量 + 元数据，第 49 课 add_documents）
    print("正在向量化并写入 Milvus ...")
    add_documents(documents)

    print(f"索引已更新，共 {len(meta)} 个片段")
    return documents


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python vector_store.py 文件路径")
        sys.exit(1)
    build_index(sys.argv[1])
