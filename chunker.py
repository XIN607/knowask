import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """一个切分片段：结构化存储内容、标题、字符数"""
    content: str        # 片段正文
    heading: str        # 所属标题路径，如 "我的测试文档 > 关于我"
    char_count: int     # 字符数


def split_by_heading(text):
    """
    按 Markdown 标题（# 开头）把文档分成若干段
    返回 [(标题路径, 内容), ...]
    """
    lines = text.split("\n")
    sections = []
    heading_stack = []   # 标题层级栈，如 [(1, "标题1"), (2, "标题2")]
    current_lines = []

    for line in lines:
        # 用正则匹配 # 开头的标题行
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            # 遇到新标题，先把上一段存起来
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    heading_path = " > ".join(h[1] for h in heading_stack) if heading_stack else "正文"
                    sections.append((heading_path, content))
                current_lines = []

            level = len(m.group(1))   # # 是1级，## 是2级
            title = m.group(2).strip()
            # 维护标题栈：弹出同级或更深的标题，压入当前标题
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            current_lines.append(line)

    # 处理最后一段
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            heading_path = " > ".join(h[1] for h in heading_stack) if heading_stack else "正文"
            sections.append((heading_path, content))

    return sections


def sliding_window_split(text, max_chars=500, overlap=100):
    """
    滑动窗口切分：每段 max_chars 字符，前后重叠 overlap 字符
    overlap 的作用：避免一句话被切断，保证上下文连续
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap   # 回退 overlap 个字符，实现重叠
    return chunks


def chunk_document(text, max_chars=500, overlap=100):
    """
    主切分函数：先按标题分段，太长的段再用滑动窗口二次切分
    返回 Chunk 列表
    """
    sections = split_by_heading(text)
    chunks = []
    for heading, content in sections:
        if len(content) <= max_chars:
            # 这段不长，直接作为一个 chunk
            chunks.append(Chunk(content=content, heading=heading, char_count=len(content)))
        else:
            # 这段太长，用滑动窗口切分
            for piece in sliding_window_split(content, max_chars, overlap):
                chunks.append(Chunk(content=piece, heading=heading, char_count=len(piece)))
    return chunks


if __name__ == "__main__":
    import sys
    from parse_doc import parse_document

    if len(sys.argv) < 2:
        print("用法: python chunker.py 文件路径")
        sys.exit(1)

    text = parse_document(sys.argv[1])
    chunks = chunk_document(text)

    print(f"文档共 {len(text)} 字符，切分为 {len(chunks)} 个片段：\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- 片段 {i} | 标题: {chunk.heading} | {chunk.char_count} 字符 ---")
        print(chunk.content[:200])
        print()
