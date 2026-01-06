import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'delta': 0.5,
    'save_steps': 10,
    'thermal_steps': 5000,
    'n_samples': 2000000,
    'bin_size': 100,
    'bootstrap_time': 1000
}


DTYPE = torch.float32


def checkerboard_metropolis_optimized(phi):
    L = CONFIG['L']
    m2 = CONFIG['m2']
    lam = CONFIG['lam']
    delta = CONFIG['delta']
    term1 = (4 + m2)

    def get_neighbor_sum(current_phi):
        up = torch.roll(current_phi, shifts=1, dims=0)
        down = torch.roll(current_phi, shifts=-1, dims=0)
        left = torch.roll(current_phi, shifts=1, dims=1)
        right = torch.roll(current_phi, shifts=-1, dims=1)
        return up + down + left + right

    coords = torch.arange(L, device=phi.device)
    i, j = torch.meshgrid(coords, coords, indexing='ij')
    mask_red = (i + j) % 2 == 0
    mask_black = (i + j) % 2 == 1

    total_accepted = 0
    total_sites = L * L

    # === Phase 1: Update Red Sites ===
    neighbor_sum = get_neighbor_sum(phi)
    change = (torch.rand_like(phi, dtype=DTYPE) - 0.5) * 2 * delta
    phi_proposal = phi + change

    # [FIX 2] 核心修正：邻居项必须乘以 2
    # test_4 的 Action 定义导致每条边贡献了两次能量
    # Metropolis 局部更新必须反映 Total Action 的变化
    S_old = term1 * phi ** 2 + lam * phi ** 4 - 2.0 * phi * neighbor_sum
    S_new = term1 * phi_proposal ** 2 + lam * phi_proposal ** 4 - 2.0 * phi_proposal * neighbor_sum
    delta_S = S_new - S_old

    random_prob = torch.rand_like(phi, dtype=DTYPE)
    accept_condition = (delta_S < 0) | (torch.exp(-delta_S) > random_prob)

    update_mask = mask_red & accept_condition
    phi = torch.where(update_mask, phi_proposal, phi)
    total_accepted += update_mask.sum().item()

    # === Phase 2: Update Black Sites ===
    neighbor_sum = get_neighbor_sum(phi)  # Recalculate neighbors
    change = (torch.rand_like(phi, dtype=DTYPE) - 0.5) * 2 * delta
    phi_proposal = phi + change

    # [FIX 2] 同样乘以 2
    S_old = term1 * phi ** 2 + lam * phi ** 4 - 2.0 * phi * neighbor_sum
    S_new = term1 * phi_proposal ** 2 + lam * phi_proposal ** 4 - 2.0 * phi_proposal * neighbor_sum
    delta_S = S_new - S_old

    random_prob = torch.rand_like(phi, dtype=DTYPE)
    accept_condition = (delta_S < 0) | (torch.exp(-delta_S) > random_prob)

    update_mask = mask_black & accept_condition
    phi = torch.where(update_mask, phi_proposal, phi)
    total_accepted += update_mask.sum().item()

    return phi, total_accepted, total_sites


# --- [FIX 3] 数据分析逻辑修正 (Global Mean) ---

def get_expected_phi(binning_ensemble):
    # 计算全系综平均（Global Mean）
    return binning_ensemble.mean()  # Scalar


def get_correlation_for_single_configuration(binning_ensemble, time_shift, space_shift):
    # 计算 <phi(x) phi(x+r)>
    ensemble_shifted = torch.roll(binning_ensemble, shifts=(time_shift, space_shift), dims=(2, 3))
    return (binning_ensemble * ensemble_shifted).mean(dim=(2, 3))


def calculate_G_t_list_inside_bin(binning_ensemble):
    G_t_list = []
    # 获取全局期望值 phi
    expected_phi = get_expected_phi(binning_ensemble)
    expected_phi_sq = expected_phi * expected_phi

    for t in range(CONFIG['L']):
        G_t_l_list = []
        for l in range(CONFIG['L']):
            conn_2_point = get_correlation_for_single_configuration(binning_ensemble, t, l)
            # 减去全局平均的平方
            conn = conn_2_point - expected_phi_sq
            G_t_l_list.append(conn)

        # 此时 list 内每个元素是 [N_bin, bin_size]
        # Stack 并在 bin_size 和 空间方向求平均
        G_t_mean_spatial = torch.stack(G_t_l_list, dim=2)  # [N_bin, bin_size, L]
        G_t_mean = G_t_mean_spatial.mean(dim=(1, 2))  # [N_bin]
        G_t_list.append(G_t_mean)

    return torch.stack(G_t_list, dim=1)  # [N_bin, Time]


def get_effective_mass(G_t):
    # 对称化处理，提高精度
    G_sym = 0.5 * (G_t + torch.flip(G_t, dims=[-1]).roll(shifts=1, dims=-1))  # G(t)应该关于 L/2 对称(周期性)
    # 但简单起见，用原始公式也可以，加 clamp
    G_plus = torch.roll(G_t, shifts=-1, dims=-1)
    G_minus = torch.roll(G_t, shifts=1, dims=-1)

    ratio = (G_plus + G_minus) / (2 * G_t + 1e-12)
    # 理论上 broken symmetry phase 应该 G(t) -> 0 (connected)，如果不为0说明 subtract 没做好
    # clamp 至少为 1
    cosh_m = torch.clamp(ratio, min=1.0000001)
    return torch.acosh(cosh_m)


def binning(data):
    N_bin = data.shape[0] // CONFIG['bin_size']
    data = data[:N_bin * CONFIG['bin_size']]
    data = data.view(N_bin, CONFIG['bin_size'], *data.shape[1:])
    return data


def bootstrap(data):
    # data: [N_bin, Time]
    N_bin = data.shape[0]
    indices = torch.randint(0, N_bin, (N_bin, CONFIG['bootstrap_time']))
    # Resample
    # [N_bin, Bootstrap, Time] -> Mean over N_bin -> [Bootstrap, Time]
    boot_samples = data[indices].mean(dim=0)
    return boot_samples


def main():
    print(f"Config: L={CONFIG['L']} [Fixed Physics]")

    # 冷启动：帮助系统快速落入有序相
    phi = torch.ones(CONFIG['L'], CONFIG['L'], dtype=DTYPE) * 0.6

    print("Start Thermalization...")
    step = 0
    while step < CONFIG['thermal_steps']:
        phi, accepted, total = checkerboard_metropolis_optimized(phi)
        step += 1

    print("Start Sampling...")
    ensemble = []
    step = 0
    total_acc = 0

    while step < CONFIG['n_samples']:
        phi, accepted, total = checkerboard_metropolis_optimized(phi)
        total_acc += accepted

        if step % CONFIG['save_steps'] == 0:
            ensemble.append(phi.clone().detach())

        step += 1
        if step % 10000 == 0:
            print(f"Step {step}/{CONFIG['n_samples']}, Acc: {total_acc / (total * 10000):.2%}")
            total_acc = 0

    # 分析
    ensemble_tensor = torch.stack(ensemble, dim=0)  # [Samples, L, L]
    print(f"Ensemble Mean Phi: {ensemble_tensor.mean():.4f} (Should be near +/- 0.6)")

    binning_data = binning(ensemble_tensor)

    # 1. 计算 G(t) (Binning后)
    G_t_bins = calculate_G_t_list_inside_bin(binning_data)  # [N_bin, L]

    # 2. Bootstrap 误差分析
    G_boot = bootstrap(G_t_bins)  # [Bootstrap, L]
    G_mean = G_boot.mean(dim=0)
    G_err = G_boot.std(dim=0)

    # 3. Effective Mass
    m_boot = get_effective_mass(G_boot)
    m_mean = m_boot.mean(dim=0)
    m_err = m_boot.std(dim=0)

    # 绘图
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    t_axis = np.arange(CONFIG['L'])
    axs[0].errorbar(t_axis, G_mean.numpy(), yerr=G_err.numpy(), fmt='-o', capsize=3)
    axs[0].set_yscale('log')
    axs[0].set_title('G(t)')

    # Mass plot (trim edges)
    axs[1].errorbar(t_axis[1:-1], m_mean.numpy()[1:-1], yerr=m_err.numpy()[1:-1], fmt='-o', color='red', capsize=3)
    axs[1].set_title('Effective Mass')
    # axs[1].set_ylim(0, 1)  # 预期质量应该在这个范围内

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()