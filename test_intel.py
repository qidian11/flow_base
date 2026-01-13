import os
import sys
from pathlib import Path


def setup_intel_environment():
    # Intel OneAPI 的默认安装根目录
    intel_root = Path(r"C:\Program Files (x86)\Intel\oneAPI")

    # 我们需要找的两个关键子目录：Compiler 和 MKL
    # 这里使用 'latest' 快捷方式，它会自动指向最新安装的版本
    candidate_paths = [
        intel_root / "compiler" / "latest" / "bin",  # 这里的 sycl7.dll 是核心
        intel_root / "mkl" / "latest" / "bin",  # MKL 库
        intel_root / "tbb" / "latest" / "bin",  # TBB 库
    ]

    print("Checking Intel OneAPI paths...")
    found_any = False
    for p in candidate_paths:
        if p.exists():
            print(f"  -> Found: {p}")
            try:
                # 1. 告诉 Python 去这里找 DLL
                os.add_dll_directory(str(p))
                # 2. 为了保险，同时也加到系统 PATH
                os.environ['PATH'] = str(p) + ";" + os.environ['PATH']
                found_any = True
            except Exception as e:
                print(f"     Error adding path: {e}")
        else:
            print(f"  -> Not found: {p}")

    if not found_any:
        print("\n[警告] 没找到 Intel OneAPI 路径！")
        print("如果你安装在 D 盘或其他位置，请手动修改脚本中的 intel_root 变量。")
        print("或者你可能只安装了显卡驱动，而没有安装 'OneAPI Base Toolkit'。")


# --- 执行环境设置 ---
setup_intel_environment()

# --- 开始测试 ---
print("\nImporting PyTorch...")
import torch
import intel_extension_for_pytorch as ipex

print(f"IPEX Version: {ipex.__version__}")

if torch.xpu.is_available():
    device_name = torch.xpu.get_device_name(0)
    print(f"\n[成功] 检测到 XPU 设备: {device_name}")

    # 简单的计算测试
    x = torch.randn(2, 2).to("xpu")
    y = torch.randn(2, 2).to("xpu")
    z = x + y
    print("计算测试通过：\n", z)
else:
    print("\n[失败] XPU 不可用，请检查驱动是否正确安装。")