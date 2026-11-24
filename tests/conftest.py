import sys
from pathlib import Path

# 让测试能 import src/ 下的模块（与 cwd 无关）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
