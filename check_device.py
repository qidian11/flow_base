import torch
import sys
import platform
import os
import subprocess


def get_size(bytes, suffix="B"):
    """
    格式化显存大小
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def check_system_hardware():
    """
    不依赖 PyTorch，直接查询操作系统硬件列表
    """
    print(f"\n{'=' * 20} 系统硬件层面检测 {'=' * 20}")
    os_type = platform.system()

    try:
        if os_type == "Windows":
            # Windows 使用 wmic 命令
            cmd = "wmic path win32_videocontroller get name,adapterram"
            result = subprocess.check_output(cmd, shell=True).decode()
            print("[Windows] 检测到的显卡设备:")
            print(result.strip())

        elif os_type == "Linux":
            # Linux 使用 lspci
            try:
                cmd = r"lspci | grep -i 'vga\|3d\|2d'"
                result = subprocess.check_output(cmd, shell=True).decode()
                print("[Linux] lspci 检测到的显卡:")
                print(result.strip())
            except:
                print("未找到 lspci 命令或无法执行。")

        elif os_type == "Darwin":
            # MacOS 使用 system_profiler
            cmd = "system_profiler SPDisplaysDataType | grep 'Chipset Model'"
            result = subprocess.check_output(cmd, shell=True).decode()
            print("[Mac] 检测到的显卡:")
            print(result.strip())

    except Exception as e:
        print(f"系统命令检测失败: {e}")


def check_pytorch_gpu():
    """
    检测 PyTorch 是否能识别 GPU
    """
    print(f"\n{'=' * 20} PyTorch 软件层面检测 {'=' * 20}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"PyTorch 版本: {torch.__version__}")

    # 1. 检测 CUDA (NVIDIA 和 AMD ROCm 都在这里)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\n✅ 检测到 GPU (CUDA/ROCm) 数量: {device_count}")

        # 判断是 NVIDIA 还是 AMD
        # AMD ROCm 在 PyTorch 中会使得 torch.version.hip 不为空
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"类型: 🔴 AMD GPU (ROCm 后端)")
            print(f"ROCm/HIP 版本: {torch.version.hip}")
        else:
            print(f"类型: 🟢 NVIDIA GPU (CUDA 后端)")
            print(f"CUDA 版本: {torch.version.cuda}")
            if torch.backends.cudnn.is_available():
                print(f"cuDNN 版本: {torch.backends.cudnn.version()}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  - 卡 {i}: {props.name}")
            print(f"    显存: {get_size(props.total_memory)}")
            print(f"    计算能力 (Capability): {props.major}.{props.minor}")

    # 2. 检测 Intel XPU (Intel Extension for PyTorch)
    # 需要安装 intel_extension_for_pytorch
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"\n✅ 检测到 🔵 Intel GPU (XPU)")
        try:
            # 不同的 IPEX 版本 API 可能不同，这里做个通用尝试
            device_count = torch.xpu.device_count()
            print(f"数量: {device_count}")
            for i in range(device_count):
                print(f"  - 卡 {i}: {torch.xpu.get_device_name(i)}")
        except:
            print("检测到 XPU 但无法获取详细信息 (驱动或库版本问题)")

    # 3. 检测 Apple Silicon (MPS)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"\n✅ 检测到 🍎 Apple Silicon GPU (MPS)")
        print("    PyTorch 正运行在 Metal Performance Shaders 上")

    # 4. 检测 DirectML (通常用于 Windows 上的 AMD/Intel 备用方案)
    else:
        # 检查是否安装了 torch-directml
        try:
            import torch_directml
            dml = torch_directml.device()
            print(f"\n⚠️ 未检测到原生 CUDA/ROCm，但检测到 DirectML 支持 (Windows AMD/Intel)")
            print(f"    设备 ID: {torch_directml.device_count()}")
            print(f"    设备名称: {torch_directml.device_name(0)}")
        except ImportError:
            print("\n❌ PyTorch 未检测到任何可用的 GPU 加速设备。")
            print("    正在使用 CPU 运行。")


if __name__ == '__main__':
    check_system_hardware()
    check_pytorch_gpu()
    print(f"\n{'=' * 60}\n")