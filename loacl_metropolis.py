import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'delta': 0.5,  # Local Metropolis 步长
    'save_steps': 1000,
    'thermal_steps': 1000,
    'n_samples': 14000,
    'bin_size': 100,      # 已移除，直接计算
    'bootstrap_time': 1000  # 已移除
}

# 确保使用双精度
DTYPE = torch.float64


def local_metropolis_sweep(phi):
    """
    Local Metropolis 算法：逐点更新
    """
    L = CONFIG['L']
    accepted_hits = 0
    total_sites = L * L

    # 提取参数避免重复查询
    m2 = CONFIG['m2']
    lam = CONFIG['lam']
    delta = CONFIG['delta']

    for i in range(L):
        for j in range(L):
            phi_old = phi[i, j]

            # 周期性边界获取邻居
            up = phi[(i - 1) % L, j]
            down = phi[(i + 1) % L, j]
            left = phi[i, (j - 1) % L]
            right = phi[i, (j + 1) % L]
            neighbor_sum = up + down + left + right

            # 提议新值
            change = (torch.rand(1, dtype=DTYPE) - 0.5) * 2 * delta
            phi_new = phi_old + change.item()

            # 计算局部 Action 变化
            S_old_local = (4 + m2) * phi_old ** 2 + lam * phi_old ** 4 - phi_old * neighbor_sum
            S_new_local = (4 + m2) * phi_new ** 2 + lam * phi_new ** 4 - phi_new * neighbor_sum

            delta_S = S_new_local - S_old_local

            # Metropolis 判据
            if delta_S < 0 or torch.rand(1).item() < torch.exp(-delta_S):
                phi[i, j] = phi_new
                accepted_hits += 1

    return phi, accepted_hits, total_sites


def calculate_G_t_fast(ensemble):
    """
    计算零动量两点关联函数 G(t)
    ensemble shape: [Samples, T, L]  (假设 T=L, 即 L x L 晶格)
    """
    # 1. 零动量投影 (Zero Momentum Projection)
    # 假设 dim=1 是时间 Time, dim=2 是空间 Space
    # 对空间求和: Phi(t) shape -> [Samples, T]
    Phi_t = ensemble.sum(dim=2)

    # 2. 计算连通部分 (Connected Part): Phi -> Phi - <Phi>
    # 先对所有样本求平均得到真空期望值 (VEV)
    vev = Phi_t.mean()
    Phi_t_fluc = Phi_t - vev

    # 3. 计算时间关联函数
    # 利用时间平移不变性，对所有起始时间 t0 进行平均
    Samples, T = Phi_t_fluc.shape
    G_t_list = []

    for t in range(T):
        # 将时间轴滚动 t
        Phi_shifted = torch.roll(Phi_t_fluc, shifts=-t, dims=1)

        # Correlator = <Phi(t0) * Phi(t0+t)>
        # 对 Samples 和 起始时间 t0 (dim=1) 同时求平均
        corr = (Phi_t_fluc * Phi_shifted).mean()
        G_t_list.append(corr)

    return torch.stack(G_t_list)


def calculate_G_t(ensemble):
    # binning_ensemble shape: [Samples, L, L] (3D)
    G_t_list = []

    for t in range(CONFIG['L']):
        G_t_l_list = []
        for l in range(CONFIG['L']):
            # conn shape: [Samples]
            conn_2_point = get_2_point_correlation(ensemble, t, l)

            # expected_phi shape: [Samples] -> mean -> scalar (or [Samples] broadcastable)
            expected_phi = get_expected_phi(ensemble)
            expected_phi_shifted = get_expected_phi(torch.roll(ensemble, shifts=(t, l), dims=(1, 2)))

            conn = conn_2_point - expected_phi * expected_phi_shifted
            G_t_l_list.append(conn)

        # shape=(time,time,space)
        G_t_mean = torch.stack(G_t_l_list).mean(dim=0)
        G_t_list.append(G_t_mean)

    # shape=(time)
    return torch.stack(G_t_list).mean(dim=(1, 2))



def get_2_point_correlation(ensemble, time_shift, space_shift):
    # 输入 shape: [Samples, Time, Space]
    # 维度修正: dims=(2,3) -> dims=(1,2)
    ensemble_shifted = torch.roll(ensemble, shifts=(time_shift, space_shift), dims=(1, 2))
    ensemble_corr = ensemble_shifted * ensemble

    # 维度修正: dims=(2,3) -> dims=(1,2)
    # 对格子 (Time, Space) 求平均，保留 [Samples]
    ensemble_corr_mean = ensemble_corr.mean(dim=0)
    return ensemble_corr_mean


def get_expected_phi(binning_ensemble):
    # 输入 shape: [Samples, Time, Space]
    # 维度修正: 保持逻辑，对整个格子求平均
    # 先对 Space/Time 求平均 -> [Samples]
    # 注意：如果此处要减去每个样本的平均值，则保留 [Samples] 维度
    return binning_ensemble.mean(dim=0)


def get_effective_mass(G_t):
    # G_t: [time]
    G_plus = torch.roll(G_t, shifts=-1, dims=-1)
    G_minus = torch.roll(G_t, shifts=1, dims=-1)

    cosh_m = (G_plus + G_minus) / (2 * G_t + 1e-10)
    cosh_m = torch.clamp(cosh_m, min=1.0)
    m_eff = torch.acosh(cosh_m)

    return m_eff


def main():
    print(f"Config: L={CONFIG['L']}, Delta={CONFIG['delta']} (Local Metropolis, No Binning)")

    # 初始化
    phi = torch.randn(CONFIG['L'], CONFIG['L'], dtype=DTYPE)

    # 预热 (Thermalization)
    print("Start Thermalization...")
    step = 0
    while step < CONFIG['thermal_steps']:
        phi, accepted, total = local_metropolis_sweep(phi)
        step += 1
        if step % 100 == 0:
            print(f"Thermal Step {step}/{CONFIG['thermal_steps']} (Acc: {accepted / total:.2f})")

    # 采样 (Sampling)
    print("Start Sampling...")
    ensemble = []
    step = 0
    total_acc = 0
    total_count = 0

    while step < CONFIG['n_samples']:
        phi, accepted, total = local_metropolis_sweep(phi)
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

    # 处理数据
    # ensemble_tensor shape: [Samples, L, L]
    ensemble_tensor = torch.stack(ensemble, dim=0)
    print(f"Ensemble shape: {ensemble_tensor.shape}")

    try:
        # 1. calculate G(t)
        # 这里传入的是 Raw Data [Samples, L, L]，函数内部已适配
        # 返回 shape: [Samples,time]
        G_t = calculate_G_t(ensemble_tensor)


        # 2. calculate effective mass
        effective_mass = get_effective_mass(G_t)

        # 截取掉首尾
        eff_mass_plot = effective_mass[1:-1]
        x_eff = np.arange(1, CONFIG['L'] - 1)

        # 绘图
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Green Function
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].plot(
            np.arange(len(G_t.numpy())),
            G_t.numpy(),
            marker='o',
            linestyle='-',
            color='blue',
            label='Raw Data'
        )
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Time Separation (t)')
        axs[0].set_ylabel('G(t) (Connected)')
        axs[0].set_title(f'2-point function (L={CONFIG["L"]})')
        axs[0].grid(True, which="both", ls="--")

        # Effective Mass
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].plot(
            x_eff,
            eff_mass_plot.numpy(),
            marker='o',
            linestyle='-',
            color='red',
            label='Raw Data'
        )
        # axs[1].set_yscale('log')
        axs[1].set_xlabel('Time Separation (t)')
        axs[1].set_ylabel('Effective Mass')
        axs[1].set_title(f'Effective Mass')
        axs[1].grid(True, which="both", ls="--")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()