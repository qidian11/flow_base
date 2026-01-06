import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'delta': 0.5,  # Metropolis step size
    'save_steps': 10,
    'thermal_steps': 10000,
    'n_samples': 100000,
    'bin_size': 100,
    'bootstrap_time': 1000
}

# 确保使用双精度
DTYPE = torch.float32


def checkerboard_metropolis_optimized(phi):
    """
    优化的红黑策略：利用切片避免无效计算
    """
    L = CONFIG['L']
    m2 = CONFIG['m2']
    lam = CONFIG['lam']
    delta = CONFIG['delta']
    term1 = (4 + m2)

    # 定义两个阶段的切片
    # 0::2 表示从0开始每隔2取一个 (偶数)
    # 1::2 表示从1开始每隔2取一个 (奇数)
    # Checkerboard patterns:
    # Set A (Red): (Even, Even) & (Odd, Odd)
    # Set B (Black): (Even, Odd) & (Odd, Even)

    # 为了简化，我们定义一个内部更新函数
    def update_subgrid(phi_val, neighbor_sum_val):
        # 仅对传入的子网格生成随机数
        change = (torch.rand_like(phi_val, dtype=DTYPE) - 0.5) * 2 * delta
        phi_new = phi_val + change

        S_old = term1 * phi_val ** 2 + lam * phi_val ** 4 - phi_val * neighbor_sum_val
        S_new = term1 * phi_new ** 2 + lam * phi_new ** 4 - phi_new * neighbor_sum_val

        d_S = S_new - S_old
        accept_prob = torch.rand_like(phi_val, dtype=DTYPE)
        mask = (d_S < 0) | (torch.exp(-d_S) > accept_prob)

        return torch.where(mask, phi_new, phi_val), mask.sum().item()

    total_accepted = 0
    total_sites = L * L

    # === Phase 1: Update Red Sites ===
    # 红点由两部分组成：(偶行,偶列) 和 (奇行,奇列)
    # 它们的邻居都在 Black 集合中，此时是固定的

    # 1.1 Update (Even, Even) - 邻居是 (E,O), (E,O), (O,E), (O,E) -> 都在 Black
    # 我们需要手动从整个 phi 中提取邻居
    # 为了避免复杂的切片索引，最简单的方法其实是先计算整个邻居和，再切片

    # 注意：为了极致优化，这里通常不使用 roll，而是直接用切片加和
    # 但为了代码可读性，我们还是计算一次全图邻居（或者只在需要的地方计算）

    # 简单优化版：还是计算全图邻居，但只对红点做 Metropolis 计算
    # 这样避免了 exp 计算的一半浪费

    neigh = torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0) + \
            torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1)

    # 创建红点掩码（这部分其实可以预计算放在外部）
    coords = torch.arange(L, device=phi.device)
    i, j = torch.meshgrid(coords, coords, indexing='ij')
    mask_red = (i + j) % 2 == 0
    mask_black = (i + j) % 2 == 1

    # 只提取红点的值和对应的邻居和
    phi_red = phi[mask_red]
    neigh_red = neigh[mask_red]

    # 更新红点
    phi_red_new, acc = update_subgrid(phi_red, neigh_red)
    phi[mask_red] = phi_red_new  # 写回
    total_accepted += acc

    # === Phase 2: Update Black Sites ===
    # 此时红点已经更新，必须重新计算邻居和
    neigh = torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0) + \
            torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1)

    phi_black = phi[mask_black]
    neigh_black = neigh[mask_black]

    phi_black_new, acc = update_subgrid(phi_black, neigh_black)
    phi[mask_black] = phi_black_new  # 写回
    total_accepted += acc

    return phi, total_accepted, total_sites


def checkerboard_metropolis_sweep(phi):
    """
    Checkerboard (Red-Black) Metropolis Algorithm.

    Instead of updating sites one by one sequentially, we divide the lattice into
    "Red" (even) and "Black" (odd) sites, similar to a chessboard.

    1. Update all Red sites simultaneously (vectorized). Their neighbors are Black sites,
       which are fixed during this step.
    2. Update all Black sites simultaneously. Their neighbors are Red sites,
       which were just updated.

    This allows for massive parallelization using PyTorch tensors.
    """
    L = CONFIG['L']
    m2 = CONFIG['m2']
    lam = CONFIG['lam']
    delta = CONFIG['delta']

    # 1. Create Masks for Red (Even) and Black (Odd) sites
    # Coordinate grid: i corresponds to row, j to column
    coords = torch.arange(L, device=phi.device)
    i, j = torch.meshgrid(coords, coords, indexing='ij')

    # Red sites: i + j is even
    mask_red = (i + j) % 2 == 0
    # Black sites: i + j is odd
    mask_black = (i + j) % 2 == 1

    total_accepted = 0
    total_sites = L * L

    # Helper function to calculate sum of neighbors using matrix rolling
    # torch.roll automatically handles periodic boundary conditions
    def get_neighbor_sum(current_phi):
        up = torch.roll(current_phi, shifts=1, dims=0)
        down = torch.roll(current_phi, shifts=-1, dims=0)
        left = torch.roll(current_phi, shifts=1, dims=1)
        right = torch.roll(current_phi, shifts=-1, dims=1)
        return up + down + left + right

    # ==========================
    # PHASE 1: Update Red Sites
    # ==========================

    # Calculate neighbors (neighbors of Red are Black, which are currently fixed)
    neighbor_sum = get_neighbor_sum(phi)

    # Propose new values for the entire grid
    # (We calculate for all, but will only apply changes to Red sites)
    change = (torch.rand_like(phi, dtype=DTYPE) - 0.5) * 2 * delta
    phi_proposal = phi + change

    # Calculate local Action (S) for old and new configurations
    # S_local = (4 + m^2) * phi^2 + lambda * phi^4 - phi * neighbor_sum
    term1 = (4 + m2)

    S_old = term1 * phi ** 2 + lam * phi ** 4 - phi * neighbor_sum
    S_new = term1 * phi_proposal ** 2 + lam * phi_proposal ** 4 - phi_proposal * neighbor_sum

    delta_S = S_new - S_old

    # Metropolis Criterion
    # Accept if delta_S < 0 OR exp(-delta_S) > random(0, 1)
    random_prob = torch.rand_like(phi, dtype=DTYPE)
    accept_condition = (delta_S < 0) | (torch.exp(-delta_S) > random_prob)

    # Combine: Must be a Red Site AND satisfy Metropolis criterion
    update_mask = mask_red & accept_condition

    # Update phi tensor only at accepted Red sites
    phi = torch.where(update_mask, phi_proposal, phi)

    # Count accepted hits for statistics
    total_accepted += update_mask.sum().item()

    # ============================
    # PHASE 2: Update Black Sites
    # ============================

    # Recalculate neighbors because Red sites have changed!
    neighbor_sum = get_neighbor_sum(phi)

    # Propose new values
    change = (torch.rand_like(phi, dtype=DTYPE) - 0.5) * 2 * delta
    phi_proposal = phi + change

    # Calculate Action change
    S_old = term1 * phi ** 2 + lam * phi ** 4 - phi * neighbor_sum
    S_new = term1 * phi_proposal ** 2 + lam * phi_proposal ** 4 - phi_proposal * neighbor_sum

    delta_S = S_new - S_old

    # Metropolis Criterion
    random_prob = torch.rand_like(phi, dtype=DTYPE)
    accept_condition = (delta_S < 0) | (torch.exp(-delta_S) > random_prob)

    # Combine: Must be a Black Site AND satisfy Metropolis criterion
    update_mask = mask_black & accept_condition

    # Update phi tensor only at accepted Black sites
    phi = torch.where(update_mask, phi_proposal, phi)

    total_accepted += update_mask.sum().item()

    return phi, total_accepted, total_sites


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
    return binning_ensemble.mean(dim=(0, 1, 2, 3))


def get_effective_mass(G_t):
    # G_t: [time]
    G_plus = torch.roll(G_t, shifts=-1, dims=-1)
    G_minus = torch.roll(G_t, shifts=1, dims=-1)

    cosh_m = (G_plus + G_minus) / (2 * G_t + 1e-10)
    cosh_m = torch.clamp(cosh_m, min=1.0)
    m_eff = torch.acosh(cosh_m)

    return m_eff



def calculate_G_t_list_inside_bin(binning_ensemble):
    G_t_list = []

    for t in range(CONFIG['L']):
        G_t_l_list = []
        for l in range(CONFIG['L']):
            # conn = get_connected_2_point_correlation(ensemble, t, l)
            conn_2_point = get_correlation_for_single_configuration(binning_ensemble, t, l)
            expected_phi = get_expected_phi(binning_ensemble)
            expected_phi_shifted = get_expected_phi(torch.roll(binning_ensemble, shifts=(t, l), dims=(2, 3)))
            conn = conn_2_point - expected_phi*expected_phi_shifted
            G_t_l_list.append(conn)
        G_t_mean = torch.stack(G_t_l_list, dim=1)
        print(f"inside calculate_G_t_list_inside_bin, "
              f"G_t_mean's shape:{G_t_mean.shape}")
        # Average drop position direction: 1 and bin_size: 2
        G_t_mean = G_t_mean.mean(dim=(1, 2))
        G_t_list.append(G_t_mean)

    return torch.stack(G_t_list, dim=1)


def get_correlation_for_single_configuration(binning_ensemble, time_shift, space_shift):
    ensemble_shifted = torch.roll(binning_ensemble, shifts=(time_shift, space_shift), dims=(2, 3))
    binning_ensemble_corr = ensemble_shifted * binning_ensemble
    binning_ensemble_corr_mean = binning_ensemble_corr.mean(dim=(2, 3))
    return binning_ensemble_corr_mean


def binning(data):
    N_bin = data.shape[0] // CONFIG['bin_size']
    data = data[:N_bin * CONFIG['bin_size']]
    data = data.view(N_bin, CONFIG['bin_size'], *data.shape[1:])
    return data


def bootstrap(data):
    N_bin = data.shape[0]
    lst = []
    for i in range(N_bin):
        idx = torch.randint(0, data.size(0), (1,))
        lst.append(data[idx.item()])
    bootstrap_ensemble = torch.stack(lst)
    # print(f"bootstrap_ensemble shape:{bootstrap_ensemble.shape}")
    return bootstrap_ensemble


def main():
    print(f"Config: L={CONFIG['L']}, Delta={CONFIG['delta']} (Checkerboard Metropolis)")

    # 初始化
    phi = torch.ones(CONFIG['L'], CONFIG['L'], dtype=DTYPE) * 0.6

    # 预热 (Thermalization)
    print("Start Thermalization...")
    step = 0
    while step < CONFIG['thermal_steps']:
        # Changed to checkerboard_metropolis_sweep
        phi, accepted, total = checkerboard_metropolis_optimized(phi)
        step += 1
        if step % 100 == 0:
            print(f"Thermal Step {step}/{CONFIG['thermal_steps']} (Acc: {accepted / total:.2f})")

    # 采样 (Sampling)
    print("Start Sampling...")
    ensemble = []
    # <phi^2>
    observable = []
    step = 0
    total_acc = 0
    total_count = 0

    while step < CONFIG['n_samples']:
        # Changed to checkerboard_metropolis_sweep
        phi, accepted, total = checkerboard_metropolis_optimized(phi)
        observable.append((phi ** 2).mean().item())
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

    #计算关联长度
    O = np.array(observable)
    O = O - O.mean()
    N = len(O)
    C = np.correlate(O, O, mode='full')
    C = C[N - 1:]  # 只要 t ≥ 0
    C = C / C[0]  # 归一化 → ρ(t)
    tau_int = 0.5
    for t in range(1, len(C)):
        if C[t] <= 0:
            break
        tau_int += C[t]
    print("tau_int =", tau_int)
    # 处理数据
    # ensemble_tensor shape: [Samples, L, L]
    ensemble_tensor = torch.stack(ensemble, dim=0)
    print(f"Ensemble shape: {ensemble_tensor.shape}")
    print(f"ensemble_mean:{ensemble_tensor.mean().item():.2e}")
    # shape=(N_bin, bin_size, time, space)
    binning_ensemble = binning(ensemble_tensor)
    try:
        # 1. calculate G(t)
        # 这里传入的是 Raw Data [Samples, L, L]，函数内部已适配
        # 返回 shape: [Samples,time]
        G_t_inside_bin = calculate_G_t_list_inside_bin(binning_ensemble)
        print(f"G_t_inside_bin's shape:{G_t_inside_bin.shape}")
        # shape=(bootsraptime, N_bin, time)
        bootstrap_tensor = []
        for i in range(CONFIG['bootstrap_time']):
            bootstrap_tensor.append(bootstrap(G_t_inside_bin))

        # green function
        # shape=(bootsraptime, time)
        bootstrap_tensor = torch.stack(bootstrap_tensor).mean(dim=1)
        print(f"bootstrap_tensor's shape:{bootstrap_tensor.shape}")
        G_error_bar = bootstrap_tensor.std(dim=0)
        y_err_g = G_error_bar.numpy()
        G_t = G_t_inside_bin.mean(dim=0)
        print(f"G_t: {G_t}")
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