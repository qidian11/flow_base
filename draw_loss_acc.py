import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib

# ==========================================
# 1. 动态加载：指定三种对比模型的代码文件与样式
# ==========================================
CONFIG_FILES = [
    {
        "module": "final_normalizing",
        "label": "Standard Normalizing Flow",
        "color": "#1f77b4",  # 蓝色
        "linestyle": "-"
    },
    {
        "module": "step0_final_normalizing",
        "label": r"Hard $Z_2$ Constraint",
        "color": "#ff7f0e",  # 橙色
        "linestyle": "--"
    },
    {
        "module": "final_normalizing_with_soft_z2",
        "label": r"Soft $Z_2$ Symmetry Penalty",
        "color": "#2ca02c",  # 绿色
        "linestyle": "-."
    }
]

# ==========================================
# 2. 📊 坐标轴范围控制中心 📊
# ==========================================
MAX_STEPS = 30000  # 全局最大读取步数
MCMC_START = 10000  # MCMC 数据截取起点
LOSS_ZOOM_START = 15000  # Loss 放大图数据截取起点

# --- 图 1: MCMC 接受率图 ---
XLIM_ACC = (10000, 30000)
YLIM_ACC = (0, 100)

# --- 图 2 & 图 3: 全局 Loss 图 ---
XLIM_LOSS_GLOBAL = (0, 30000)

# --- 图 3: 局部放大 Loss 图 (画中画) ---
XLIM_LOSS_ZOOM = (15000, 30000)
YLIM_LOSS_ZOOM = None  # 可以根据实际数据调整

# ==========================================
# 3. 自动导入配置
# ==========================================
EXPERIMENTS = []
for item in CONFIG_FILES:
    try:
        mod = importlib.import_module(item["module"])
        EXPERIMENTS.append({
            "config": mod.CONFIG,
            "label": item["label"],
            "color": item["color"],
            "linestyle": item["linestyle"]
        })
        print(f"✅ 成功导入配置: {item['module']}")
    except ImportError as e:
        print(f"❌ 导入失败 {item['module']}: {e}")


def smooth_data(data, window_size=500):
    if len(data) < window_size: return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')


def load_data_from_config(config, module_name):
    data = {"loss": None, "mcmc_steps": None, "acc": None}
    pt_path = config.get('checkpoint_path', '')
    npz_path = config.get('observables_save_path', '')

    print(f"\n🔍 [诊断] 正在从模块 '{module_name}' 载入数据...")

    if not pt_path:
        print("  ⚠️ 未检测到 'checkpoint_path' 键")
    elif not os.path.exists(pt_path):
        print(f"  ❌ 找不到 PT 文件: {pt_path}")
    else:
        try:
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
            if 'history_loss' in ckpt:
                data["loss"] = np.array(ckpt['history_loss'])[:MAX_STEPS]
                print(f"  ✅ Loss 加载成功 (起点: {data['loss'][0]:.2f})")
            else:
                print("  ⚠️ PT 文件中没有 'history_loss'")
        except Exception as e:
            print(f"  ❌ PT 文件崩溃: {e}")

    if not npz_path:
        print("  ⚠️ 未检测到 'observables_save_path' 键")
    elif not os.path.exists(npz_path):
        print(f"  ❌ 找不到 NPZ 文件: {npz_path}")
    else:
        try:
            npz_data = np.load(npz_path)
            if 'steps' in npz_data and 'acc' in npz_data:
                steps, acc = npz_data['steps'], npz_data['acc']
                mask = (steps <= MAX_STEPS) & (steps >= MCMC_START)
                data["mcmc_steps"] = steps[mask]
                data["acc"] = acc[mask]
                print(f"  ✅ MCMC 加载成功 (点数: {len(data['mcmc_steps'])})")
        except Exception as e:
            print(f"  ❌ NPZ 文件崩溃: {e}")

    return data


# ==========================================
# 4. 绘图主程序
# ==========================================
def plot_results():
    loaded_data = []
    for item, exp in zip(CONFIG_FILES, EXPERIMENTS):
        data = load_data_from_config(exp["config"], item["module"])
        loaded_data.append((exp, data))

    print("\n📊 正在生成图表...")

    # 降采样步长：专门为了解决密集数据点导致虚线连成实线的问题
    DS_STEP = 50

    # ---------------------------------------------------------
    # 图 1: MCMC 接受率
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    has_acc_data = False
    for exp, data in loaded_data:
        if data["mcmc_steps"] is not None and len(data["mcmc_steps"]) > 0:
            has_acc_data = True
            ax1.plot(data["mcmc_steps"], data["acc"] * 100,
                     color=exp["color"], linestyle=exp["linestyle"], marker='o',
                     markersize=4, label=exp["label"], linewidth=1.5, alpha=0.9)

    ax1.set_title("MCMC Acceptance Rate (10k - 30k Steps)", fontsize=14)
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Acceptance Rate (%)", fontsize=12)
    ax1.axhline(y=78.0, color='gray', linestyle=':', linewidth=1.5, label="Target (78%)")

    if XLIM_ACC is not None: ax1.set_xlim(XLIM_ACC)
    if YLIM_ACC is not None: ax1.set_ylim(YLIM_ACC)
    ax1.grid(True, ls="--", alpha=0.6)
    if has_acc_data: ax1.legend(fontsize=10, loc="lower right")
    fig1.tight_layout()
    fig1.savefig("plot_1_acc_comparison.png")

    # ---------------------------------------------------------
    # 图 2: Loss 全局图 (独立)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=300)
    has_loss_data = False
    for i, (exp, data) in enumerate(loaded_data):
        if data["loss"] is not None and len(data["loss"]) > 0:
            has_loss_data = True
            iters = np.arange(1, len(data["loss"]) + 1)
            ax2.plot(iters, data["loss"], color=exp["color"], alpha=0.15)

            smooth_loss = smooth_data(data["loss"])
            s_iters = np.arange(len(smooth_loss)) + (len(data["loss"]) - len(smooth_loss)) // 2

            # ====== 新增降采样切片 [::DS_STEP] ======
            ax2.plot(s_iters[::DS_STEP], smooth_loss[::DS_STEP], color=exp["color"],
                     linestyle=exp["linestyle"], label=exp["label"], linewidth=2)

            start_loss = data["loss"][0]
            ax2.plot(0, start_loss, marker='*', color=exp["color"], markersize=10, markeredgecolor='black', zorder=5)

            v_offset = [1.5, 0, -1.5][i] if i < 3 else 0
            ax2.text(800, start_loss + v_offset, f"Start: {start_loss:.1f}", color=exp["color"],
                     fontsize=10, fontweight='bold', va='center')

    ax2.set_title("Training Loss Comparison (0 - 30k Steps)", fontsize=14)
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel(r"Loss $= \mathcal{L}(\theta)$", fontsize=14)

    if XLIM_LOSS_GLOBAL is not None: ax2.set_xlim(XLIM_LOSS_GLOBAL)

    ax2.set_ylim(bottom=-70)

    ax2.ticklabel_format(style='plain', axis='y')
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    if has_loss_data: ax2.legend(fontsize=10, loc="lower left")
    fig2.tight_layout()
    fig2.savefig("plot_2_loss_global_comparison.png")

    # ---------------------------------------------------------
    # 图 3: 高级排版 (全局 Loss 嵌套 画中画指示图)
    # ---------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=300)
    axins = ax4.inset_axes([0.45, 0.45, 0.5, 0.45])

    for i, (exp, data) in enumerate(loaded_data):
        if data["loss"] is not None and len(data["loss"]) > 0:
            # 1. 绘制主图 (全局)
            iters = np.arange(1, len(data["loss"]) + 1)
            ax4.plot(iters, data["loss"], color=exp["color"], alpha=0.15)

            smooth_loss = smooth_data(data["loss"])
            s_iters = np.arange(len(smooth_loss)) + (len(data["loss"]) - len(smooth_loss)) // 2

            # ====== 新增降采样切片 [::DS_STEP] ======
            ax4.plot(s_iters[::DS_STEP], smooth_loss[::DS_STEP], color=exp["color"],
                     linestyle=exp["linestyle"], label=exp["label"], linewidth=2)

            start_loss = data["loss"][0]
            ax4.plot(0, start_loss, marker='*', color=exp["color"], markersize=10, markeredgecolor='black', zorder=5)

            v_offset = [1.5, 0, -1.5][i] if i < 3 else 0
            ax4.text(800, start_loss + v_offset, f"Start: {start_loss:.1f}", color=exp["color"],
                     fontsize=10, fontweight='bold', va='center')

            # 2. 绘制画中画 (局部放大)
            if len(data["loss"]) > LOSS_ZOOM_START:
                zoom_loss = data["loss"][LOSS_ZOOM_START:MAX_STEPS]
                zoom_iters = np.arange(LOSS_ZOOM_START + 1, LOSS_ZOOM_START + len(zoom_loss) + 1)
                axins.plot(zoom_iters, zoom_loss, color=exp["color"], alpha=0.2)

                smooth_zoom = smooth_data(zoom_loss, window_size=200)
                s_zoom_iters = np.arange(len(smooth_zoom)) + LOSS_ZOOM_START + (len(zoom_loss) - len(smooth_zoom)) // 2

                # ====== 核心修复：画中画平滑线降采样 ======
                axins.plot(s_zoom_iters[::DS_STEP], smooth_zoom[::DS_STEP], color=exp["color"],
                           linestyle=exp["linestyle"], linewidth=2)

    ax4.set_title("Training Loss with Local Zoom", fontsize=14)
    ax4.set_xlabel("Steps", fontsize=12)
    ax4.set_ylabel(r"Loss $= \mathcal{L}(\theta)$", fontsize=14)

    if XLIM_LOSS_GLOBAL is not None: ax4.set_xlim(XLIM_LOSS_GLOBAL)

    ax4.set_ylim(bottom=-70)

    ax4.ticklabel_format(style='plain', axis='y')
    ax4.grid(True, which="both", ls="--", alpha=0.4)

    if has_loss_data:
        ax4.legend(fontsize=10, loc="lower left", framealpha=0.9)

    if XLIM_LOSS_ZOOM is not None: axins.set_xlim(XLIM_LOSS_ZOOM)
    if YLIM_LOSS_ZOOM is not None: axins.set_ylim(YLIM_LOSS_ZOOM)

    axins.ticklabel_format(style='plain', axis='y')
    axins.tick_params(labelleft=True, labelbottom=True)
    axins.grid(True, ls="--", alpha=0.4)

    ax4.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)

    fig4.tight_layout()
    fig4.savefig("plot_3_loss_inset_comparison.png")

    print("\n🎉 3 张图表均已生成完毕！")
    print("   -> 图1: MCMC 接受率对比 (plot_1_acc_comparison.png)")
    print("   -> 图2: Loss 全局对比图 (plot_2_loss_global_comparison.png)")
    print("   -> 图3: 画中画复合对比图 (plot_3_loss_inset_comparison.png)")


if __name__ == "__main__":
    plot_results()