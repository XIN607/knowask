"""项目核心数据模型（Pydantic）。

用类型明确的数据结构替代散落在各处的 dict，让检索、切分各环节的
字段有 IDE 提示、有校验，改字段名时能立即发现遗漏。
"""
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """一个切分片段：正文、所属标题路径、字符数。"""

    content: str = Field(description="片段正文")
    heading: str = Field(default="正文", description="所属标题路径，如 '我的文档 > 关于我'")
    char_count: int = Field(default=0, ge=0, description="字符数")


class SearchResult(BaseModel):
    """一次检索返回的一个片段。"""

    rank: int = Field(ge=1, description="排名（从 1 开始）")
    score: float = Field(description="相关性分数")
    heading: str = Field(description="所属标题路径")
    content: str = Field(description="片段正文")
    idx: int = Field(ge=0, description="片段在元数据列表中的下标")
    source: str = Field(default="未知来源", description="来源文件名")


class CheckResult(BaseModel):
    """拒答判断结果（第 38 课 PydanticOutputParser 的目标模型）。"""

    found: bool = Field(description="资料中是否能找到答案")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0-1")
