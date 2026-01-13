import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_device(prefer="auto"):
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if prefer == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")

    return torch.device("cpu")


device = get_device(prefer="xpu")

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'tao': 1.18,
    'leap_frog_step': 10,
    'save_steps': 10,
    'thermal_steps': 1000,
    'n_samples': 1000000,
    'bin_size': 10,
    'bootstrap_time': 1000,

}


DTYPE = torch.float64



def calculate_action(phi):
    # phi shape: [L, L]
    # 这里的维度变成了 0 和 1
    phi_up = torch.roll(phi, shifts=-1, dims=0)
    phi_down = torch.roll(phi, shifts=1, dims=0)
    phi_right = torch.roll(phi, shifts=1, dims=1)
    phi_left = torch.roll(phi, shifts=-1, dims=1)

    # 动能项 (离散拉普拉斯算子部分)
    kinetic_term = 4 * phi * phi - phi * (phi_right + phi_left + phi_up + phi_down)
    potential_term = CONFIG['m2'] * phi * phi + CONFIG['lam'] * phi ** 4

    # 对整个 grid 求和，得到标量 Action
    action = (kinetic_term + potential_term).sum()
    return action


def calculate_kinetic_energy(p):
    # p shape: [L, L]
    return 0.5 * torch.sum(p ** 2)


def calculate_hamiltonian(phi, p):
    action = calculate_action(phi)
    kinetic_energy = calculate_kinetic_energy(p)
    return kinetic_energy + action


def get_force(phi):
    # phi shape: [L, L]
    phi_up = torch.roll(phi, shifts=-1, dims=0)
    phi_down = torch.roll(phi, shifts=1, dims=0)
    phi_right = torch.roll(phi, shifts=1, dims=1)
    phi_left = torch.roll(phi, shifts=-1, dims=1)

    # 导数计算
    return -(8 * phi - 2 * (phi_up + phi_down + phi_right + phi_left)
             + 2 * CONFIG['m2'] * phi + 4 * CONFIG['lam'] * phi ** 3)


def get_velocity(p):
    return p


def leap_frog(phi, p, tao):
    epsilon = tao / CONFIG['leap_frog_step']

    # Half step for momentum
    force = get_force(phi)
    p_new = p + (epsilon / 2) * force

    phi_new = phi.clone()

    # Full steps
    for i in range(CONFIG['leap_frog_step']):
        phi_new = phi_new + epsilon * get_velocity(p_new)

        # NaN 检查
        if torch.isnan(phi_new).any():
            print("NaN detected in leap frog")
            return phi_new, p_new

        if i != CONFIG['leap_frog_step'] - 1:
            p_new = p_new + epsilon * get_force(phi_new)

    # Final half step for momentum
    p_new = p_new + (epsilon / 2) * get_force(phi_new)

    if torch.isnan(phi_new).any():
        print("NaN detected in leap frog")
    return phi_new, p_new


def HMC_step_single_chain(phi, tao):
    """
    输入 phi shape: [L, L]
    """
    # 1. get new p, shape = [L, L]
    p = torch.randn(CONFIG['L'], CONFIG['L'], dtype=DTYPE, device=phi.device)

    hamiltonian_old = calculate_hamiltonian(phi, p)

    # 2. Leapfrog
    phi_new, p_new = leap_frog(phi, p, tao)

    # 3. inverse leapfrog
    phi_inverse, p_inverse = leap_frog(phi_new, -p_new, tao)
    delta_phi = phi - phi_inverse
    delta_p = p + p_inverse
    hamiltonian_inverse = calculate_hamiltonian(phi_inverse, p_inverse)

    delta_inverse_H = hamiltonian_inverse - hamiltonian_old
    # --- NaN check ---
    if torch.isnan(phi_new).any() or torch.isinf(phi_new).any():
        return phi, delta_inverse_H, delta_phi, delta_p, False, False

    hamiltonian_new = calculate_hamiltonian(phi_new, p_new)

    # calculate delta H
    delta_H = hamiltonian_new - hamiltonian_old

    if torch.isnan(delta_H) or torch.isinf(delta_H):
        return phi, delta_H, delta_inverse_H, delta_phi, delta_p, False, False

    # 3. Metropolis 接受/拒绝
    prob = torch.exp(-delta_H).item()
    rand_num = torch.rand(1).item()

    if rand_num < prob:
        return phi_new, delta_H, delta_inverse_H, delta_phi, delta_p, True, True
    else:
        return phi, delta_H, delta_inverse_H, delta_phi, delta_p, True, False


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
        # print(f"inside calculate_G_t_list_inside_bin, "
        #       f"G_t_mean's shape:{G_t_mean.shape}")
        G_t_mean = G_t_mean.mean(dim=(1, 2))
        G_t_list.append(G_t_mean)

    return torch.stack(G_t_list, dim=1)  # shape: N_bin, t


def get_correlation_for_single_configuration(binning_ensemble, time_shift, space_shift):
    ensemble_shifted = torch.roll(binning_ensemble, shifts=(time_shift, space_shift), dims=(2, 3))
    binning_ensemble_corr = ensemble_shifted * binning_ensemble
    binning_ensemble_corr_mean = binning_ensemble_corr.mean(dim=(2, 3))
    return binning_ensemble_corr_mean


def get_correlation_from_single_configuration(binning_ensemble):
    return binning_ensemble.main(dims=(0, 1))


def get_expected_phi(binning_ensemble):
    expected_phi = binning_ensemble.mean(dim=(0, 1))
    expected_phi = expected_phi.mean(dim=(0, 1))
    return expected_phi


def get_effective_mass(G_t):
    """
    输入 G_t: [..., L]
    输出 m_eff: [..., L]
    """
    # 1. 利用 roll 处理周期性边界，计算全量 L 个点
    G_plus = torch.roll(G_t, shifts=-1, dims=-1)
    G_minus = torch.roll(G_t, shifts=1, dims=-1)

    # 2. 核心公式
    # 加上 1e-10 防止除零
    cosh_m = (G_plus + G_minus) / (2 * G_t + 1e-10)

    # 3. Clamp 防止 NaN (保底)
    cosh_m = torch.clamp(cosh_m, min=1.0)

    # 4. 计算反双曲余弦
    m_eff = torch.acosh(cosh_m)

    return m_eff


def binning(data):
    N_bin = data.shape[0] // CONFIG['bin_size']
    data = data[:N_bin * CONFIG['bin_size']]
    data = data.view(N_bin, CONFIG['bin_size'], *data.shape[1:])
    return data


def bootstrap(data,bootsrtrap_time):
    N_bin = data.shape[0]
    lst = []
    for _ in range(bootsrtrap_time):
        idx = torch.randint(0, N_bin, (N_bin,))
        lst.append(data[idx].mean())
    return torch.stack(lst)

    for i in range(N_bin):
        idx = torch.randint(0, data.size(0), (1,))
        lst.append(data[idx.item()])
    bootstrap_ensemble = torch.stack(lst)
    # print(f"bootstrap_ensemble shape:{bootstrap_ensemble.shape}")
    return bootstrap_ensemble


def main():
    print(f"Config: L={CONFIG['L']}, Tao={CONFIG['tao']} (No Batch/Channel dims)")

    # 初始化：直接生成 [L, L]
    phi = torch.zeros(CONFIG['L'], CONFIG['L'], dtype=DTYPE, device=device)
    delta_H_list = []
    delta_inverse_hamiltonian_list = []
    delta_phi_list = []
    delta_p_list = []
    # 预热 (Thermalization)
    print("Start Thermalization...")
    step = 0
    while step < CONFIG['thermal_steps']:
        phi, delta_H, inverse_hamiltonian, delta_phi, delta_p, success, accepted = HMC_step_single_chain(phi, CONFIG['tao'])

        if not success:
            print(f"Thermal Step {step}: NaN detected! Retrying...")
            if step == 0:
                phi = torch.zeros(CONFIG['L'], CONFIG['L'], dtype=DTYPE, device=device)
            continue

        step += 1
        if step % 100 == 0:
            print(f"Thermal Step {step}/{CONFIG['thermal_steps']}")

    # 采样 (Sampling)
    print("Start Sampling...")
    ensemble = []
    step = 0
    accept_count = 0

    while step < CONFIG['n_samples']:
        phi, delta_H, inverse_hamiltonian, delta_phi, delta_p, success, accepted = HMC_step_single_chain(phi, CONFIG['tao'])

        if not success:
            print(f"Sample Step {step}: NaN detected! Retrying...")
            continue

        delta_H_list.append(delta_H)
        delta_inverse_hamiltonian_list.append(inverse_hamiltonian)
        delta_phi_list.append(delta_phi)
        delta_p_list.append(delta_p)
        if accepted:
            accept_count += 1

        if step % CONFIG['save_steps'] == 0:
            # detach 并存入列表
            ensemble.append(phi.clone().detach().cpu())

        step += 1
        if step % 100 == 0:
            print(f"Sampling Step {step}/{CONFIG['n_samples']}, Accept Ratio: {accept_count / step * 100:.2f}%")

    # 处理数据
    # ensemble_tensor shape: [Samples, L, L]
    ensemble_tensor = torch.stack(ensemble, dim=0)
    print(f"Ensemble shape: {ensemble_tensor.shape}")

    delta_inverse_hamiltonian_tensor = torch.tensor(delta_inverse_hamiltonian_list)
    delta_H_tensor = torch.stack(delta_H_list, dim=0)
    delta_phi_list_tensor = torch.stack(delta_phi_list, dim=0)
    delta_p_list_tensor = torch.stack(delta_p_list, dim=0)


    binning_delta_H_tensor = binning(delta_H_tensor)
    # mean in bin
    binning_delta_H_tensor = binning_delta_H_tensor.mean(dim=1)
    bootstrap_delta_H_tensor = bootstrap(binning_delta_H_tensor, CONFIG['bootstrap_time'])
    delta_H_error = bootstrap_delta_H_tensor.std(dim=0)
    print(f'expect of delta_H:{torch.mean(delta_H_tensor).item():.2e}')
    print(f"{torch.mean(binning_delta_H_tensor):.5f}({int(delta_H_error * 1e5):02d})")
    delta_H_exp_tensor = torch.exp(delta_H_tensor)
    delta_H_exp_std = torch.std(delta_H_exp_tensor).item()
    delta_H_exp_error = delta_H_exp_std / torch.sqrt(torch.tensor(delta_H_exp_tensor.numel(), device=delta_H_exp_tensor.device))
    print(f'expect of delta_H_exp:{torch.mean(delta_H_exp_tensor).item():.2e}')
    print(f'delta_H_exp_error:{delta_H_exp_error:.2e}')

    # print(f"ensemble_mean:{ensemble_tensor.mean().item():.2e}")
    # print(f'delta_inverse_hamiltonian_list:{delta_inverse_hamiltonian_list}')
    # delta_inverse_hamiltonian_tensor = torch.cat(delta_inverse_hamiltonian_list)
    # y = delta_inverse_hamiltonian_tensor  # 你的 1D torch tensor
    y = np.array([x.item() for x in delta_inverse_hamiltonian_list])
    x = torch.arange(len(y))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("index")
    plt.ylabel("delta hamiltonian")
    plt.show()

    # delta_phi
    y = np.array([x.item() for x in delta_phi_list_tensor.mean(dim=(1,2)).cpu().numpy()])
    x = torch.arange(len(y))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("index")
    plt.ylabel("delta phi")
    plt.show()

    # delta_p
    y = np.array([x.item() for x in delta_p_list_tensor.mean(dim=(1,2)).cpu().numpy()])
    x = torch.arange(len(y))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("index")
    plt.ylabel("delta p")
    plt.show()

    # shape=(N_bin, bin_size, time, space)
    binning_ensemble = binning(ensemble_tensor)
    phi_to_1 = binning_ensemble.mean()
    phi_to_2 = (binning_ensemble * binning_ensemble).mean()
    phi_to_3 = (binning_ensemble **3).mean()
    phi_to_4 = (binning_ensemble ** 4).mean()
    phi_to_5 = (binning_ensemble ** 5).mean()
    print("delta_H: {:.2e}".format(delta_H_tensor.mean().item()))
    print("expectation of exp delta_h: {:.2e}".format(torch.exp(-delta_H_tensor).mean().item()))
    print(f"phi_to_1:{phi_to_1:.2e}")
    print(f"phi_to_2:{phi_to_2:.2e}")
    print(f"phi_to_3:{phi_to_3:.2e}")
    print(f"phi_to_4:{phi_to_4:.2e}")
    print(f"phi_to_5:{phi_to_5:.2e}")
    try:
        # shape: N_bin, t
        G_t_inside_bin = calculate_G_t_list_inside_bin(binning_ensemble)
        print(f"G_t_inside_bin's shape:{G_t_inside_bin.shape}")
        # shape=(bootsraptime, N_bin, time)

        # for i in range(CONFIG['bootstrap_time']):
        #     bootstrap_tensor.append(bootstrap(G_t_inside_bin))
        # shape=(bootsraptime, time)

        bootstrap_tensor = bootstrap(binning_ensemble, CONFIG['bootstrap_time'])
        # green function
        # shape=(bootsraptime, time)
        # bootstrap_tensor = torch.stack(bootstrap_tensor).mean(dim=1)
        print(f"bootstrap_tensor's shape:{bootstrap_tensor.shape}")
        G_error_bar = bootstrap_tensor.std(dim=0)
        y_err_g = G_error_bar.numpy()
        G_t = G_t_inside_bin.mean(dim=0)

        # effective mass
        # shape=(time)
        effective_mass = get_effective_mass(G_t)
        effective_mass = effective_mass[1:-1]
        bootstrap_effective_mass = get_effective_mass(bootstrap_tensor)
        bootstrap_effective_mass = bootstrap_effective_mass[:, 1:-1]
        e_m_error_bar = bootstrap_effective_mass.std(dim=0)
        y_err_m = e_m_error_bar.numpy()
        # plt
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # green function
        # axs[0].plot(G_t.numpy(), np.arange(CONFIG['L']), marker='o', linestyle='-')
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].errorbar(
            np.arange(len(G_t.numpy())),
            G_t.numpy(),
            yerr=y_err_g,
            fmt='-o',  # 格式字符串: '-'代表连线, 'o'代表画点
            color='blue',  # 线和点的颜色
            ecolor='purple',  # error bar 的颜色 (设为不同颜色方便看清)
            capsize=4,  # 误差棒两端“帽子”横线的宽度
            elinewidth=1.5,  # 误差棒线条的粗细
            label='Lattice Data'  # 图例标签
        )
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Time Separation (t)')
        axs[0].set_ylabel('G(t) (Connected)')
        axs[0].set_title(f'2-point function (L={CONFIG["L"]}, tao={CONFIG["tao"]})')
        axs[0].grid(True, which="both", ls="--")

        # effective mass
        # axs[1].plot(effective_mass.numpy(), np.arange(1, CONFIG['L']-1), marker='o', linestyle='-')
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].errorbar(
            np.arange(len(e_m_error_bar.numpy())),
            effective_mass.numpy(),
            yerr=y_err_m,
            fmt='-o',  # 格式字符串: '-'代表连线, 'o'代表画点
            color='blue',  # 线和点的颜色
            ecolor='purple',  # error bar 的颜色 (设为不同颜色方便看清)
            capsize=4,  # 误差棒两端“帽子”横线的宽度
            elinewidth=1.5,  # 误差棒线条的粗细
            label='Lattice Data'  # 图例标签
        )
        axs[1].set_xlabel('Time Separation (t)')
        axs[1].set_ylabel('effective mass')
        axs[1].set_title(f'effective mass (L={CONFIG["L"]}, tao={CONFIG["tao"]})')
        # axs[1].grid(True, which="both", ls="--")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
