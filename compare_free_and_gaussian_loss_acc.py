import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 大道至简：直接 Import 所有配置
# ==========================================
from final_gaussian_normalizing import CONFIG as cfg_gauss
from final_gaussian_normalizing_64_18 import CONFIG as cfg_gauss_large
from final_normalizing_32_abs import CONFIG as cfg_free_abs
from final_normalizing import CONFIG as cfg_free
from final_normalizing_with_sample_mean import CONFIG as cfg_free_opt

# 🌟 核心修改：全部标明具体网络架构 (ch, L)，明确 m^2_free 和 c 🌟
EXPERIMENTS = [
    {
        "config": cfg_gauss,
        "label": r"Standard Gaussian ($c=0$, 32 ch, 12 L)",
        "color": "#d62728",  # 红
        "linestyle": "-"
    },
    {
        "config": cfg_gauss_large,
        "label": r"Standard Gaussian ($c=0$, 64 ch, 18 L)",
        "color": "#ff7f0e",  # 橙
        "linestyle": "--"
    },
    {
        "config": cfg_free_abs,
        "label": r"Free-Field ($m_{free}^2=4.0, c=0$, 32 ch, 12 L)",
        "color": "#9467bd",  # 紫
        "linestyle": "-"
    },
    {
        "config": cfg_free,
        # 修正：同样是最优的 0.6005，但是均值为 0
        "label": r"Free-Field ($m_{free}^2 \approx 0.6005, c=0$, 32 ch, 12 L)",
        "color": "#1f77b4",  # 蓝
        "linestyle": "-"
    },
    {
        "config": cfg_free_opt,
        # 标明 c=1.0
        "label": r"Free-Field ($m_{free}^2 \approx 0.6005, \mathbf{c=1.0}$, 32 ch, 12 L)",
        "color": "#2ca02c",  # 绿
        "linestyle": "-"
    }
]

# ==========================================
# 2. 数据读取 (精准使用 Config 中的绝对路径)
# ==========================================
MAX_STEPS = 30000
MCMC_START = 10000  # MCMC 严控从 10000 步开始


def smooth_data(data, window_size=500):
    if len(data) < window_size: return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')


def load_data_from_config(config):
    data = {"loss": None, "mcmc_steps": None, "acc": None}

    pt_path = config['checkpoint_path']
    npz_path = config['observables_save_path']

    print(f"\n🔍 正在读取: {config.get('type', 'Unknown Type')}")

    # 1. 读 Loss
    if os.path.exists(pt_path):
        try:
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
            if 'history_loss' in ckpt:
                data["loss"] = np.array(ckpt['history_loss'])[:MAX_STEPS]
                print(f"✅ Loss 加载成功 (起点值: {data['loss'][0]:.2f})")
        except Exception as e:
            print(f"⚠️ PT 读取失败: {e}")
    else:
        print(f"❌ 找不到 PT 文件: {pt_path}")

    # 2. 读 接受率
    if os.path.exists(npz_path):
        try:
            npz_data = np.load(npz_path)
            steps = npz_data['steps']
            acc = npz_data['acc']

            # 严格卡点 10000 到 30000 步
            mask = (steps <= MAX_STEPS) & (steps >= MCMC_START)
            data["mcmc_steps"] = steps[mask]
            data["acc"] = acc[mask]
            print(f"✅ 接受率加载成功 (有效点数: {len(data['mcmc_steps'])})")
        except Exception as e:
            print(f"⚠️ NPZ 读取失败: {e}")
    else:
        print(f"❌ 找不到 NPZ 文件: {npz_path}")

    return data


# ==========================================
# 3. 绘图主程序
# ==========================================
def plot_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    for exp in EXPERIMENTS:
        data = load_data_from_config(exp["config"])
        color = exp["color"]
        label = exp["label"]
        linestyle = exp["linestyle"]

        # --- 绘制 Loss ---
        if data["loss"] is not None and len(data["loss"]) > 0:
            iters = np.arange(1, len(data["loss"]) + 1)
            ax1.plot(iters, data["loss"], color=color, alpha=0.15)

            smooth_loss = smooth_data(data["loss"])
            s_iters = np.arange(len(smooth_loss)) + (len(data["loss"]) - len(smooth_loss)) // 2
            ax1.plot(s_iters, smooth_loss, color=color, linestyle=linestyle, label=label, linewidth=2)

            # ⭐️ 标出起点 Loss，动态偏移 ⭐️
            init_loss = data["loss"][0]
            ax1.plot(1, init_loss, marker='*', color=color, markersize=12, zorder=10, markeredgecolor='black',
                     markeredgewidth=0.5)

            # 错开相近起点的值，避免文字重叠
            offset_y = 0
            if init_loss > 2000 and "64 ch" in label:
                offset_y = -15

            ax1.annotate(f"Start: {init_loss:.1f}",
                         xy=(1, init_loss),
                         xytext=(15, offset_y),
                         textcoords="offset points",
                         color=color,
                         fontsize=10,
                         fontweight='bold',
                         va='center',
                         zorder=10)

        # --- 绘制 接受率 ---
        if data["mcmc_steps"] is not None and len(data["mcmc_steps"]) > 0:
            ax2.plot(data["mcmc_steps"], data["acc"] * 100,
                     color=color, linestyle=linestyle, marker='o', markersize=4,
                     label=label, linewidth=1.5, alpha=0.9)

    # ==========================
    # 左图设置: Loss
    # ==========================
    ax1.set_title("Training Loss History (First 30,000 Steps)", fontsize=14, pad=15)
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Loss (Negative Log-Likelihood + Action)", fontsize=12)
    ax1.set_xlim(0, MAX_STEPS)

    # 采用 symlog 使得 0 到 -64 范围也能被线性完美展示，同时兼容 2700 的巨大起点
    ax1.set_yscale("symlog", linthresh=100.0)

    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.9)

    # ==========================
    # 右图设置: Acceptance Rate
    # ==========================
    ax2.set_title("MCMC Acceptance Rate (10,000 to 30,000 Steps)", fontsize=14, pad=15)
    ax2.set_xlabel("Iterations", fontsize=12)
    ax2.set_ylabel("Acceptance Rate (%)", fontsize=12)

    ax2.set_xlim(MCMC_START, MAX_STEPS)
    ax2.set_ylim(0, 100)

    # 🚨 修正：目标接受率 70% 🚨
    ax2.axhline(y=70.0, color='gray', linestyle=':', linewidth=1.5, label="Target Acceptance (70%)")

    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("paper_figure_30k_comparison_final.png", bbox_inches='tight')
    print("\n🎉 绘图完成！图表已保存为: paper_figure_30k_comparison_final.png")
    plt.show()


if __name__ == "__main__":
    plot_results()