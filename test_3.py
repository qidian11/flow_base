import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'delta': 0.5,
    'save_steps': 10,
    'thermal_steps': 1000,
    'n_samples': 3000,  # 注意：这会导致总共只有 14000/1000 = 14 个样本！
    'bin_size': 100,
    'bootstrap_time': 1000
}

DTYPE = torch.float64


def check_tensor(tensor, name, step=None):
    """
    通用数据检查函数
    """
    if tensor is None:
        return

    # 检查 NaN
    if torch.isnan(tensor).any():
        msg = f"!!! NaN detected in [{name}]"
        if step is not None:
            msg += f" at step {step}"
        print(f"\n{msg}")
        # 打印部分数据用于调试
        print(f"Sample data: {tensor.flatten()[:5]}")
        sys.exit(1)  # 遇到错误直接停止，方便定位

    # 检查 Inf
    if torch.isinf(tensor).any():
        msg = f"!!! Inf detected in [{name}]"
        if step is not None:
            msg += f" at step {step}"
        print(f"\n{msg}")
        sys.exit(1)


def local_metropolis_sweep(phi):
    L = CONFIG['L']
    accepted_hits = 0
    total_sites = L * L
    m2 = CONFIG['m2']
    lam = CONFIG['lam']
    delta = CONFIG['delta']

    for i in range(L):
        for j in range(L):
            phi_old = phi[i, j]

            # 周期性边界
            up = phi[(i - 1) % L, j]
            down = phi[(i + 1) % L, j]
            left = phi[i, (j - 1) % L]
            right = phi[i, (j + 1) % L]
            neighbor_sum = up + down + left + right

            change = (torch.rand(1, dtype=DTYPE) - 0.5) * 2 * delta
            phi_new = phi_old + change.item()

            S_old_local = (4 + m2) * phi_old ** 2 + lam * phi_old ** 4 - phi_old * neighbor_sum
            S_new_local = (4 + m2) * phi_new ** 2 + lam * phi_new ** 4 - phi_new * neighbor_sum

            delta_S = S_new_local - S_old_local

            if delta_S < 0 or torch.rand(1).item() < torch.exp(-delta_S):
                phi[i, j] = phi_new
                accepted_hits += 1

    return phi, accepted_hits, total_sites


def calculate_G_t_fast(ensemble):
    """
    建议使用这个优化的函数，计算更稳定
    """
    print("Computing G(t)...")
    check_tensor(ensemble, "Ensemble input to G(t)")

    # [Samples, T, L] -> 空间求和得到 Phi(t) -> [Samples, T]
    Phi_t = ensemble.sum(dim=2)

    # 减去真空期望值 (VEV)
    vev = Phi_t.mean()
    Phi_t_fluc = Phi_t - vev

    Samples, T = Phi_t_fluc.shape
    G_t_list = []

    for t in range(T):
        # 时间平移相关
        Phi_shifted = torch.roll(Phi_t_fluc, shifts=-t, dims=1)
        corr = (Phi_t_fluc * Phi_shifted).mean()  # 对所有样本和基准时间求平均
        G_t_list.append(corr)

    G_t = torch.stack(G_t_list)
    check_tensor(G_t, "G(t) Result")
    return G_t


def get_effective_mass(G_t):
    # G_t: [time]
    check_tensor(G_t, "Input to Effective Mass")

    G_plus = torch.roll(G_t, shifts=-1, dims=-1)
    G_minus = torch.roll(G_t, shifts=1, dims=-1)

    # 检查除零风险
    denominator = 2 * G_t
    if torch.any(torch.abs(denominator) < 1e-12):
        print("Warning: G(t) is very close to zero, causing division instability.")

    ratio = (G_plus + G_minus) / (denominator + 1e-15)

    # cosh(m) must >= 1
    # check NaN
    if torch.any(ratio < 1.0):
        print("Warning: cosh(m) < 1 detected! This causes NaNs in acosh.")
        print(f"Indices where ratio < 1: {torch.where(ratio < 1.0)}")
        # 强制截断，防止程序崩溃，但这意味着物理结果在这里由于噪声已不可信
        ratio = torch.clamp(ratio, min=1.000001)

    m_eff = torch.acosh(ratio)
    check_tensor(m_eff, "Effective Mass Result")
    return m_eff


def main():
    print(f"Config: L={CONFIG['L']}, N_Samples={CONFIG['n_samples']}")

    phi = torch.randn(CONFIG['L'], CONFIG['L'], dtype=DTYPE)
    check_tensor(phi, "Initial Phi")

    print("Start Thermalization...")
    step = 0
    while step < CONFIG['thermal_steps']:
        phi, accepted, total = local_metropolis_sweep(phi)
        step += 1

        if step % 100 == 0:
            check_tensor(phi, "Phi during thermalization", step)

    # --- 3. 采样 ---
    print("Start Sampling...")
    ensemble = []
    step = 0
    total_acc = 0
    total_count = 0

    while step < CONFIG['n_samples']:
        phi, accepted, total = local_metropolis_sweep(phi)

        # [CHECK] 每次更新后检查
        # 为了性能，可以改为每 100 步检查一次
        # check_tensor(phi, "Phi during sampling", step)

        total_acc += accepted
        total_count += total

        if step % CONFIG['save_steps'] == 0:
            ensemble.append(phi.clone().detach())

        step += 1
        if step % 1000 == 0:
            ratio = total_acc / total_count if total_count > 0 else 0
            print(f"Sampling Step {step}/{CONFIG['n_samples']}, Accept Ratio: {ratio * 100:.2f}%")
            total_acc = 0
            total_count = 0
            # 定期检查当前场
            check_tensor(phi, "Phi (periodic check)", step)

    # --- 4. 数据处理 ---
    if len(ensemble) == 0:
        print("Error: No samples collected! Check n_samples vs save_steps.")
        return

    ensemble_tensor = torch.stack(ensemble, dim=0)
    print(f"Ensemble shape: {ensemble_tensor.shape}")
    check_tensor(ensemble_tensor, "Final Ensemble Tensor")

    try:
        # 使用 fast 版本，原版逻辑可能在维度缩减上有问题
        G_t = calculate_G_t_fast(ensemble_tensor)

        # 有效质量
        effective_mass = get_effective_mass(G_t)

        # 截取掉首尾 (t=0 和 t=T/2 附近通常由源/汇效应主导)
        # 注意有效质量通常只在中间平台期 (Plateau) 有效
        eff_mass_plot = effective_mass[1:-1]
        x_eff = np.arange(1, CONFIG['L'] - 1)

        # --- 绘图 ---
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Green Function
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].plot(
            np.arange(len(G_t.numpy())),
            G_t.numpy(),
            marker='o',
            linestyle='-',
            color='blue',
            label='G(t)'
        )
        axs[0].set_yscale('log')
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('G(t)')
        axs[0].set_title(f'2-point Correlator')
        axs[0].grid(True, which="both", ls="--", alpha=0.5)

        # Effective Mass
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].plot(
            x_eff,
            eff_mass_plot.numpy(),
            marker='o',
            linestyle='-',
            color='red',
            label='m_eff'
        )
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('m_eff')
        axs[1].set_title(f'Effective Mass')
        axs[1].grid(True, which="both", ls="--", alpha=0.5)

        # 限制 y 轴范围，防止极值破坏视图
        if not torch.isnan(eff_mass_plot).all():
            median_mass = torch.nanmedian(eff_mass_plot).item()
            axs[1].set_ylim(median_mass * 0.5, median_mass * 1.5)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()