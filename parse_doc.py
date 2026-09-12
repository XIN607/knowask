from pathlib import Path

def read_text_file(path):
    """读取 .txt/.md 文件 (纯文本,Python自带能力)"""
    return Path(path).read_text(encoding="utf-8")

def read_pdf(path):
    """读取 PDF 文件: 逐页抽取文字,拼成一篇"""
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)

def read_word(path):
    """读取 Word文件: 抽取每个段落文字"""
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_document(path):
    """根据文件后缀自动选择解析方式，统一返回纯文本"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in(".txt",".md",".markdown"):
        return read_text_file(path)
    if suffix ==".pdf":
        return read_pdf(path)
    if suffix in(".docx",):
        return read_word(path)
    raise ValueError(f"不支持的文件类型：{suffix}")

if __name__ =="__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python parse_doc.py 文件路径")
        sys.exit(1)
    text = parse_document(sys.argv[1])
    print(f"解析成功！共{len(text)}个字符，前 500 字如下：\n")
    print(text[:500])