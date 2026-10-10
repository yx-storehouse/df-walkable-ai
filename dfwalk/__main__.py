"""python -m dfwalk：与 dfwalk 命令完全等价。"""

import sys

from .cli import main

if __name__ == "__main__":  # Windows 下 DataLoader 的 spawn 子进程导入本模块时不会再次执行
    sys.exit(main())
