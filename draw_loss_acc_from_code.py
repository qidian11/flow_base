import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib

# ==========================================
# 1. 动态加载：在这里指定你的代码文件列表与样式
# ==========================================
CONFIG_FILES = [
    {
        "module": "final_gaussian_normalizing",
        "label": r"Standard Gaussian ($c=0$, 32 ch, 12 L)",
        "color": "#d62728",
        "linestyle": "--"
    },
    {
        "module": "final_normalizing",
        "label": r"Free-Field ($m_{free}^2 \approx 0.6005, c=0, 32 ch, 12 L$)",
        "color": "#1f77b4",
        "linestyle": "-"
    }
    # 往下继续添加你的配置...
]

# ==========================================
# 2. 📊 坐标轴范围控制中心 📊
# ==========================================
# [提示] 如果设为 None，Matplotlib 会根据数据自动计算该轴范围。

# --- 图 1: MCMC 接受率图 ---
XLIM_ACC = (10000, 30000)  # X轴显示范围 (默认显示 MCMC 阶段)
YLIM_ACC = (0, 100)  # Y轴显示范围

# --- 图 2: 全局 Loss 图 ---
XLIM_LOSS_GLOBAL = (0, 30000)  # X轴显示范围
YLIM_LOSS_GLOBAL = None  # Y轴显示范围 (设为None自动启用 symlog 兼容巨大跨度)

# --- 图 3: 局部放大 Loss 图 ---
XLIM_LOSS_ZOOM = (15000, 30000)  # X轴显示范围
YLIM_LOSS_ZOOM = (-62.5, -61)  # Y轴显示范围 (建议硬编码以获得最佳对比度)

# ==========================================
# 3. 数据读取控制 (决定把多少数据读进内存)
# ==========================================
MAX_STEPS = 30000  # 全局最大读取步数
MCMC_START = 10000  # MCMC 数据截取起点
LOSS_ZOOM_START = 15000  # Loss 放大图数据截取起点

# ==========================================
# 4. 自动导入配置
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
# 5. 绘图主程序
# ==========================================
def plot_results():
    loaded_data = []
    for item, exp in zip(CONFIG_FILES, EXPERIMENTS):
        data = load_data_from_config(exp["config"], item["module"])
        loaded_data.append((exp, data))

    print("\n📊 正在生成图表...")

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

    ax1.set_title("MCMC Acceptance Rate (>10k Steps)", fontsize=14)
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Acceptance Rate (%)", fontsize=12)
    ax1.axhline(y=70.0, color='gray', linestyle=':', linewidth=1.5, label="Target (70%)")

    if XLIM_ACC is not None: ax1.set_xlim(XLIM_ACC)
    if YLIM_ACC is not None: ax1.set_ylim(YLIM_ACC)
    ax1.grid(True, ls="--", alpha=0.6)
    if has_acc_data: ax1.legend(fontsize=10, loc="lower right")
    fig1.tight_layout()
    fig1.savefig("plot_1_acc.png")

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
            ax2.plot(s_iters, smooth_loss, color=exp["color"], linestyle=exp["linestyle"], label=exp["label"],
                     linewidth=2)

            init_loss = data["loss"][0]
            ax2.plot(1, init_loss, marker='*', color=exp["color"], markersize=12, zorder=10, markeredgecolor='black',
                     markeredgewidth=0.5)
            offset_y = (i * 14) - (len(loaded_data) * 7)
            ax2.annotate(f"Start: {init_loss:.1f}", xy=(1, init_loss), xytext=(15, offset_y),
                         textcoords="offset points", color=exp["color"], fontsize=10, fontweight='bold', va='center',
                         zorder=10)

    ax2.set_title("Global Training Loss", fontsize=14)
    ax2.set_xlabel("Iterations", fontsize=12)
    ax2.set_ylabel(r"Loss $= D_{\mathrm{KL}}$", fontsize=14)

    if XLIM_LOSS_GLOBAL is not None: ax2.set_xlim(XLIM_LOSS_GLOBAL)
    if YLIM_LOSS_GLOBAL is not None:
        ax2.set_ylim(YLIM_LOSS_GLOBAL)
    else:
        ax2.set_yscale("symlog", linthresh=100.0)

    ax2.grid(True, which="both", ls="--", alpha=0.4)
    if has_loss_data: ax2.legend(fontsize=10, loc="upper right")
    fig2.tight_layout()
    fig2.savefig("plot_2_loss_global.png")

    # ---------------------------------------------------------
    # 图 3: Loss 放大图 (独立)
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=300)
    for exp, data in loaded_data:
        if data["loss"] is not None and len(data["loss"]) > LOSS_ZOOM_START:
            zoom_loss = data["loss"][LOSS_ZOOM_START:MAX_STEPS]
            zoom_iters = np.arange(LOSS_ZOOM_START + 1, LOSS_ZOOM_START + len(zoom_loss) + 1)
            ax3.plot(zoom_iters, zoom_loss, color=exp["color"], alpha=0.2)

            smooth_zoom = smooth_data(zoom_loss, window_size=200)
            s_zoom_iters = np.arange(len(smooth_zoom)) + LOSS_ZOOM_START + (len(zoom_loss) - len(smooth_zoom)) // 2
            ax3.plot(s_zoom_iters, smooth_zoom, color=exp["color"], linestyle=exp["linestyle"], label=exp["label"],
                     linewidth=2)

    ax3.set_title("Zoomed Training Loss", fontsize=14)
    ax3.set_xlabel("Iterations", fontsize=12)
    ax3.set_ylabel(r"Loss $= D_{\mathrm{KL}}$", fontsize=14)

    if XLIM_LOSS_ZOOM is not None: ax3.set_xlim(XLIM_LOSS_ZOOM)
    if YLIM_LOSS_ZOOM is not None: ax3.set_ylim(YLIM_LOSS_ZOOM)
    ax3.grid(True, ls="--", alpha=0.4)
    if has_loss_data: ax3.legend(fontsize=10, loc="upper right")
    fig3.tight_layout()
    fig3.savefig("plot_3_loss_zoomed.png")

    # ---------------------------------------------------------
    # 图 4: 高级排版 (全局 Loss 嵌套 画中画指示图)
    # ---------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=300)
    # 将放大图放在主图右上方: [x, y, width, height]
    axins = ax4.inset_axes([0.45, 0.45, 0.5, 0.45])

    for i, (exp, data) in enumerate(loaded_data):
        if data["loss"] is not None and len(data["loss"]) > 0:
            # 1. 绘制主图 (全局)
            iters = np.arange(1, len(data["loss"]) + 1)
            ax4.plot(iters, data["loss"], color=exp["color"], alpha=0.15)

            smooth_loss = smooth_data(data["loss"])
            s_iters = np.arange(len(smooth_loss)) + (len(data["loss"]) - len(smooth_loss)) // 2
            ax4.plot(s_iters, smooth_loss, color=exp["color"], linestyle=exp["linestyle"], label=exp["label"],
                     linewidth=2)

            init_loss = data["loss"][0]
            ax4.plot(1, init_loss, marker='*', color=exp["color"], markersize=12, zorder=10, markeredgecolor='black',
                     markeredgewidth=0.5)
            offset_y = (i * 14) - (len(loaded_data) * 7)
            ax4.annotate(f"Start: {init_loss:.1f}", xy=(1, init_loss), xytext=(15, offset_y),
                         textcoords="offset points", color=exp["color"], fontsize=10, fontweight='bold', va='center',
                         zorder=10)

            # 2. 绘制画中画 (局部放大)
            if len(data["loss"]) > LOSS_ZOOM_START:
                zoom_loss = data["loss"][LOSS_ZOOM_START:MAX_STEPS]
                zoom_iters = np.arange(LOSS_ZOOM_START + 1, LOSS_ZOOM_START + len(zoom_loss) + 1)
                axins.plot(zoom_iters, zoom_loss, color=exp["color"], alpha=0.2)

                smooth_zoom = smooth_data(zoom_loss, window_size=200)
                s_zoom_iters = np.arange(len(smooth_zoom)) + LOSS_ZOOM_START + (len(zoom_loss) - len(smooth_zoom)) // 2
                # 画中画不需要加 label，避免图例重复
                axins.plot(s_zoom_iters, smooth_zoom, color=exp["color"], linestyle=exp["linestyle"], linewidth=2)

    ax4.set_title("Global Training Loss with Local Zoom", fontsize=14)
    ax4.set_xlabel("Iterations", fontsize=12)
    ax4.set_ylabel(r"Loss $= D_{\mathrm{KL}}$", fontsize=14)

    if XLIM_LOSS_GLOBAL is not None: ax4.set_xlim(XLIM_LOSS_GLOBAL)
    if YLIM_LOSS_GLOBAL is not None:
        ax4.set_ylim(YLIM_LOSS_GLOBAL)
    else:
        ax4.set_yscale("symlog", linthresh=100.0)
    ax4.grid(True, which="both", ls="--", alpha=0.4)

    if has_loss_data:
        # 为了不挡住右侧的画中画，图例移到左下角
        ax4.legend(fontsize=10, loc="lower left", framealpha=0.9)

    if XLIM_LOSS_ZOOM is not None: axins.set_xlim(XLIM_LOSS_ZOOM)
    if YLIM_LOSS_ZOOM is not None: axins.set_ylim(YLIM_LOSS_ZOOM)
    axins.tick_params(labelleft=True, labelbottom=True)
    axins.grid(True, ls="--", alpha=0.4)

    # 画出连接框与指示线
    ax4.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)

    fig4.tight_layout()
    fig4.savefig("plot_4_loss_inset.png")

    print("\n🎉 4 张图片均已生成完毕！")
    print("   -> 图1: MCMC 接受率 (plot_1_acc.png)")
    print("   -> 图2: Loss 全局图 (plot_2_loss_global.png)")
    print("   -> 图3: Loss 放大图 (plot_3_loss_zoomed.png)")
    print("   -> 图4: 画中画复合图 (plot_4_loss_inset.png)")
    plt.show()


if __name__ == "__main__":
    plot_results()