import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


def print_phi_powers_table(hmc_data, cnn_data):
    """
    在控制台打印排版好的 phi 幂次对比表
    """
    hmc_mean, hmc_err = hmc_data['phi_pow_mean'], hmc_data['phi_pow_err']
    cnn_mean, cnn_err = cnn_data['phi_pow_mean'], cnn_data['phi_pow_err']

    print("\n" + "=" * 70)
    print(f"{'Observable':<12} | {'HMC (Base)':<25} | {'Prior CNN (Flow)':<25}")
    print("-" * 70)

    for i in range(5):
        power = i + 1
        hmc_str = f"{hmc_mean[i]:.6f} ± {hmc_err[i]:.6f}"
        cnn_str = f"{cnn_mean[i]:.6f} ± {cnn_err[i]:.6f}"
        print(f"phi^{power:<9} | {hmc_str:<25} | {cnn_str:<25}")
    print("=" * 70 + "\n")


def plot_observables_comparison(hmc_file, cnn_file):
    if not os.path.exists(hmc_file) or not os.path.exists(cnn_file):
        print("❌ 找不到指定的数据文件，请检查文件名和路径！")
        return

    # 加载数据
    hmc_data = np.load(hmc_file)
    cnn_data = np.load(cnn_file)
    L = int(hmc_data['L'])  # 假设两者 L 一致

    # 打印表格
    print_phi_powers_table(hmc_data, cnn_data)

    # ================= 绘图部分 =================
    # 采用高能物理常用的黑白对比色系 (HMC 黑圈, CNN 蓝钻或紫方块)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    t_axis = np.arange(L)

    # --- 左图：连通格林函数 G_c(t) ---
    axs[0].set_box_aspect(1 / 1.4)

    # HMC 基准线 (黑色, 圆圈)
    axs[0].errorbar(t_axis, hmc_data['G_t_mean'], yerr=hmc_data['G_t_err'],
                    fmt='-o', color='black', ecolor='black', capsize=4, elinewidth=1.5,
                    label='HMC', markersize=5, mfc='none')  # mfc='none' 让圆圈空心，更清爽

    # Prior CNN 模型线 (蓝色, 菱形)
    axs[0].errorbar(t_axis, cnn_data['G_t_mean'], yerr=cnn_data['G_t_err'],
                    fmt='-D', color='darkblue', ecolor='purple', capsize=4, elinewidth=1.5,
                    label='Prior CNN', markersize=5)

    axs[0].set_yscale('log')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_xlabel('Time Separation (t)')
    axs[0].set_ylabel(r'$\tilde{G}_c(0, t)$')
    axs[0].set_title('Connected Two-point Green\'s Function')
    axs[0].grid(True, which="both", ls="--", alpha=0.3)
    axs[0].legend()

    # --- 右图：有效质量 m_eff(t) ---
    axs[1].set_box_aspect(1 / 1.4)
    slice_idx = slice(1, -1)  # 截掉 t=0 和 t=L-1 的边界失真
    t_axis_m = t_axis[slice_idx]

    axs[1].errorbar(t_axis_m, hmc_data['m_eff_mean'][slice_idx], yerr=hmc_data['m_eff_err'][slice_idx],
                    fmt='-o', color='black', ecolor='black', capsize=4, elinewidth=1.5,
                    label='HMC', markersize=5, mfc='none')

    axs[1].errorbar(t_axis_m, cnn_data['m_eff_mean'][slice_idx], yerr=cnn_data['m_eff_err'][slice_idx],
                    fmt='-D', color='darkblue', ecolor='purple', capsize=4, elinewidth=1.5,
                    label='Prior CNN', markersize=5)

    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_xlabel('Time Separation (t)')
    axs[1].set_ylabel(r'$m_p^{\rm eff}(t)$')
    axs[1].set_title(f'Effective Pole Mass Comparison (L={L})')
    axs[1].grid(True, which="major", ls="--", alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    # plt.savefig(f"Observables_Compare_L{L}.pdf", bbox_inches='tight') # 如果你想保存PDF可以取消注释
    plt.show()


if __name__ == '__main__':
    # 将你的文件路径填在这里
    HMC_FILE = "HMC_observables_result_L14_double_precision_True.npz"
    CNN_FILE = "Prior_CNN_observables_result_L14_double_precision_True.npz"

    plot_observables_comparison(HMC_FILE, CNN_FILE)