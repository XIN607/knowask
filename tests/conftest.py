"""pytest 公共配置：让测试能导入项目根目录下的 app 包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
