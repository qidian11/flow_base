import torch
# 修正后的导入
from ptflops import get_model_complexity_info
from cnn_res_net import FlowModel, CONFIG


def profile_model():
    # 1. 实例化模型并切换到评估模式
    CONFIG['batch_size'] = 1
    model = FlowModel(CONFIG).cpu()
    model.eval()

    # 2. 定义输入尺寸 (Channels, Height, Width)
    # 对于你的标量场，通道是 1
    input_shape = (1, CONFIG['L'], CONFIG['L'])

    # 3. 计算复杂度
    # print_per_layer_stat: 显示每一层的详细开销（如各个 ResBlock）
    # as_strings: 将结果转换为易读的格式（如 16.66 GMac）
    macs, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # 理论 FLOPs 换算
    # 1 MAC (Multiply-Accumulate) 通常等于 2 FLOPs
    mac_val = float(macs.split()[0])
    unit = macs.split()[1].lower()

    multiplier = 1
    if 'g' in unit:
        multiplier = 2  # 如果是 GMac，对应 2 GFLOPs
    elif 'm' in unit:
        multiplier = 0.002  # 如果是 MMac

    print(f"理论总 FLOPs (1 MAC ≈ 2 FLOPs): {mac_val * multiplier:.2f} GFLOPs")


if __name__ == "__main__":
    profile_model()