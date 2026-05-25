import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 实验配置区 (易扩展设计)
# ==========================================
# 后续如果有新的模型（比如不同的网络层数、或者非多尺度版本的 CNN），
# 只需要在这个列表中新增一个字典配置即可。
EXPERIMENTS = [
    {
        "name": "Z_2 Symmetry Flow",  # 图例中显示的名称
        # 请替换为实际跑出来的最新或最好的 checkpoint 文件路径
        "filepath": "latest_single_modal_z_2_symmetry_shared_coupling_model_double_precision_False_14_coupling_layers_6_multi_k_3_dil_1_depth_2_layers_6cnn_hidden_layers_4_hidden_channels_16_iterations_25000.pt",
        "color": "#1f77b4",  # 科技蓝
        "linestyle": "-"
    },
    {
        "name": "No Z_2 Symmetry Flow",
        # 请替换为实际跑出来的最新或最好的 checkpoint 文件路径
        "filepath": "latest_aligned_prior_cnn_double_precision_False_14_coupling_layers_6_multi_k_3_dil_1_depth_2_layers_6cnn_hidden_layers_4_hidden_channels_16_iterations_25000.pt",
        "color": "#ff7f0e",  # 活力橙
        "linestyle": "-"
    }
    # 扩展示例：
    # {
    #     "name": "Standard CNN (No Multi-Scale)",
    #     "filepath": "path/to/another_model.pt",
    #     "color": "#2ca02c",
    #     "linestyle": "--"
    # }
]


# ==========================================
# 2. 核心数据提取与处理
# ==========================================
def extract_loss_from_checkpoint(filepath):
    """从 PyTorch Checkpoint 中安全提取 Loss 历史"""
    if not os.path.exists(filepath):
        print(f"⚠️ 警告: 找不到文件 {filepath}")
        return None

    try:
        # map_location='cpu' 确保即使在没有 GPU 的机器上也能读取进行绘图
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        history_loss = checkpoint.get('history_loss', [])
        return np.array(history_loss)
    except Exception as e:
        print(f"❌ 读取 {filepath} 失败: {e}")
        return None


def calculate_ema(data, alpha=0.05):
    """
    计算指数移动平均 (EMA)。
    参数 alpha 对应你训练代码中的 0.05 (0.95 * ema_loss + 0.05 * loss_val)
    """
    if len(data) == 0: return data

    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = (1 - alpha) * ema[i - 1] + alpha * data[i]
    return ema


# ==========================================
# 3. 学术风绘图模块 (新增 x_range 控制)
# ==========================================
def plot_loss_comparison(experiments, alpha=0.01, x_range=None, y_range=None, save_fig=False,save_name="loss_comparison.png"):
    """
    绘制对比图：
    x_range: 传入元组 (xmin, xmax) 手动限制 x 轴范围，例如 (10000, 25000)
    y_range: 传入元组 (ymin, ymax) 手动限制 y 轴范围，例如 (-10, 50)
    """
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'legend.fontsize': 12, 'figure.dpi': 300, 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linestyle': '--'
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_plot_count = 0

    for exp in experiments:
        if exp["filepath"] is None:
            continue

        raw_loss = extract_loss_from_checkpoint(exp["filepath"])
        if raw_loss is None or len(raw_loss) == 0:
            continue

        valid_plot_count += 1
        iterations = np.arange(1, len(raw_loss) + 1)
        ema_loss = calculate_ema(raw_loss, alpha=alpha)

        ax.plot(iterations, raw_loss, color=exp["color"], alpha=0.15, linewidth=0.5)
        ax.plot(iterations, ema_loss, color=exp["color"], linestyle=exp["linestyle"],
                linewidth=2.0, label=f"{exp['name']} (EMA)")

    if valid_plot_count == 0:
        print("❌ 没有任何有效的实验数据可供绘制，请检查权重路径。")
        plt.close()
        return

    # ==========================================
    # 🌟 坐标轴范围控制逻辑
    # ==========================================
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    ax.set_title("Training Action Loss Comparison")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$\mathcal{L} = \log q(z) - \log \det J + S(\phi)$")
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    plt.tight_layout()

    # 修改最后保存图片的部分
    if save_fig:
        plt.savefig(save_name, bbox_inches='tight')
        print(f"✅ 绘图已保存至 {save_name}")
    plt.show()


if __name__ == "__main__":
    # ==========================================
    # 🌟 统一调用入口，在这里控制平滑度与显示范围
    # ==========================================
    plot_loss_comparison(
        EXPERIMENTS,
        alpha=0.05,
        x_range=(0, 15000),  # 示例：直接切掉前 5000 步的剧烈震荡
        y_range=(-65, 0),   # 如果纵轴也被前期异常值拉伸得很厉害，可以取消这行注释来限制 y 轴
        save_fig=True
    )
    # ==========================================
    # 🌟 第二张图：单独的局部放大图 (看后期收敛细节)
    # ==========================================
    print("正在生成图 2：局部放大细节...")
    plot_loss_comparison(
        EXPERIMENTS,
        alpha=0.05,
        x_range=(2000, 15000),  # 截取最后 10000 步
        y_range=(-65, -55),  # 缩小 y 轴范围，死扣最后期的微弱差异 (数值请根据实际 loss 调整)
        save_fig=True,
        save_name="zoomed_loss_comparison.png"  # 第二张图的名字
    )