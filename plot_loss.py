import torch
import matplotlib.pyplot as plt


def plot_loss_comparison():
    # 两个模型的文件名
    file_abs = "latest_shared_trunk_prior_cnn8_m_abs_dp_False_L14_c6_d2_TrCh8x6_Ly3x6_trk_3_dil_1_Sh8x6L1x6_Th8x6L1x6_iter_100000.pt"
    file_free = "latest_shared_trunk_prior_cnn8_m_free_dp_False_L14_c6_d2_TrCh8x6_Ly3x6_trk_3_dil_1_Sh8x6L1x6_Th8x6L1x6_iter_100000.pt"

    # 设置截止步数
    max_steps = 6000

    # 1. 提取 m^2=4 的 loss 数据并截断
    try:
        data_abs = torch.load(file_abs, map_location='cpu')
        loss_abs = data_abs.get('history_loss', [])[:max_steps]
    except Exception as e:
        print(f"❌ 读取 {file_abs} 失败: {e}")
        return

    # 2. 提取 m^2 ≈ 0.6005 的 loss 数据并截断
    try:
        data_free = torch.load(file_free, map_location='cpu')
        loss_free = data_free.get('history_loss', [])[:max_steps]
    except Exception as e:
        print(f"❌ 读取 {file_free} 失败: {e}")
        return

    print(f"✅ 成功加载并截断数据！")
    print(f"   m^2=4 实际绘图步数: {len(loss_abs)}")
    print(f"   m^2 ≈ 0.6005 实际绘图步数: {len(loss_free)}")

    # 3. 创建画布，1行2列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ------------------ 图1：全局 Loss 对比 (0 ~ 6000步) ------------------
    ax1.plot(loss_abs, label=r"$m^2=4$", color='#1f77b4', alpha=0.8, linewidth=1)
    ax1.plot(loss_free, label=r"$m^2 \approx 0.6005$", color='#ff7f0e', alpha=0.8, linewidth=1)

    # 🌟 新增：在图1标明第0步的 Loss
    if len(loss_abs) > 0:
        ax1.plot(0, loss_abs[0], marker='o', markersize=6, color='#1f77b4')
        ax1.annotate(f"Step 0: {loss_abs[0]:.4f}",
                     xy=(0, loss_abs[0]),
                     xytext=(15, 15), textcoords='offset points',  # 向右上偏移15像素
                     color='#1f77b4', fontweight='bold', fontsize=10,
                     arrowprops=dict(arrowstyle="->", color='#1f77b4', alpha=0.7))

    if len(loss_free) > 0:
        ax1.plot(0, loss_free[0], marker='o', markersize=6, color='#ff7f0e')
        ax1.annotate(f"Step 0: {loss_free[0]:.4f}",
                     xy=(0, loss_free[0]),
                     xytext=(15, -20), textcoords='offset points',  # 向右下偏移20像素，防止与上面重叠
                     color='#ff7f0e', fontweight='bold', fontsize=10,
                     arrowprops=dict(arrowstyle="->", color='#ff7f0e', alpha=0.7))

    ax1.set_title(f"Global Loss History (Up to {max_steps} Iterations)", fontsize=14, pad=10)
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ------------------ 图2：局部放大 Loss 对比 (2000 ~ 6000步) ------------------
    start_idx = 2000
    if len(loss_abs) > start_idx and len(loss_free) > start_idx:
        ax2.plot(range(start_idx, len(loss_abs)), loss_abs[start_idx:],
                 label=r"$m^2=4$", color='#1f77b4', alpha=0.8, linewidth=1)
        ax2.plot(range(start_idx, len(loss_free)), loss_free[start_idx:],
                 label=r"$m^2 \approx 0.6005$", color='#ff7f0e', alpha=0.8, linewidth=1)
        ax2.set_title(f"Zoomed-in Loss History (Iteration {start_idx} to {max_steps})", fontsize=14, pad=10)
        ax2.set_xlabel("Iterations", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, f"Not enough iterations\n(Total < {start_idx})",
                 ha='center', va='center', fontsize=12)
        ax2.set_title("Zoomed-in Loss History")

    # 4. 调整布局并保存
    plt.tight_layout()
    save_path = "loss_comparison_6000.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📷 历史曲线对比图已保存为: {save_path}")

    # 5. 弹出窗口预览
    plt.show()


if __name__ == "__main__":
    plot_loss_comparison()