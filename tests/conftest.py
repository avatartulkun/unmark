import sys
from pathlib import Path

# 让测试能 import 到同目录上一层的 unwatermark / server。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
