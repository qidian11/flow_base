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


device = get_device(prefer="cpu")
print(f"🔥 当前测试设备: {device}")

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'tao': 1.18,
    'leap_frog_step': 10,
    'save_steps': 10,
    'thermal_steps': 1000,
    'n_samples': 120000,
    'bin_size': 100,
    'bootstrap_time': 2000,
    'batch_size': 128,
}


DTYPE = torch.float32



def calculate_action(phi):
    # phi shape: [batchsize, L, L]
    # 这里的维度变成了 0 和 1
    phi_up = torch.roll(phi, shifts=-1, dims=1)
    phi_down = torch.roll(phi, shifts=1, dims=1)
    phi_right = torch.roll(phi, shifts=1, dims=2)
    phi_left = torch.roll(phi, shifts=-1, dims=2)

    # 动能项 (离散拉普拉斯算子部分)
    kinetic_term = 4 * phi * phi - phi * (phi_right + phi_left + phi_up + phi_down)
    potential_term = CONFIG['m2'] * phi * phi + CONFIG['lam'] * phi ** 4

    # 对整个 grid 求和，得到标量 Action
    action = (kinetic_term + potential_term).sum(dim=(1,2))
    return action


def calculate_kinetic_energy(p):
    # p shape: [L, L]
    return 0.5 * torch.sum(p ** 2, dim=(1,2))


def calculate_hamiltonian(phi, p):
    action = calculate_action(phi)
    kinetic_energy = calculate_kinetic_energy(p)
    return kinetic_energy + action


def get_force(phi):
    # phi shape: [L, L]
    phi_up = torch.roll(phi, shifts=-1, dims=1)
    phi_down = torch.roll(phi, shifts=1, dims=1)
    phi_right = torch.roll(phi, shifts=1, dims=2)
    phi_left = torch.roll(phi, shifts=-1, dims=2)

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


def HMC_step(phi, tao):
    # 1. 为每条链独立生成动量
    p = torch.randn_like(phi)
    h_old = calculate_hamiltonian(phi, p)

    # 2. Leapfrog 演化
    phi_new, p_new = leap_frog(phi, p, tao)

    # 3. 计算反向演化（用于可逆性检查 delta_phi, delta_p）
    phi_inv, p_inv = leap_frog(phi_new, -p_new, tao)
    delta_phi = phi - phi_inv  # [Batch, L, L]
    delta_p = p + p_inv        # [Batch, L, L]

    h_inv = calculate_hamiltonian(phi_inv, p_inv)
    delta_inverse_H = h_inv - h_old # [Batch]

    # 4. 判定接受
    h_new = calculate_hamiltonian(phi_new, p_new)
    delta_H = h_new - h_old # [Batch]

    # --- NaN/Inf 检查 ---
    # 如果某条链数值爆炸，我们标记 success 为 False
    success = ~torch.isnan(delta_H).any() and ~torch.isinf(delta_H).any()
    if not success:
        return phi, delta_H, delta_inverse_H, delta_phi, delta_p, False, 0.0

    # 5. Metropolis 接受/拒绝 (Per-chain)
    prob = torch.exp(-delta_H.clamp(max=50)) # 防止溢出
    rand_num = torch.rand_like(prob)
    accepted_mask = rand_num < prob # [Batch] 的布尔掩码

    # 更新 phi：接受的用 phi_new，拒绝的保留原 phi
    phi_next = torch.where(accepted_mask.view(-1, 1, 1), phi_new, phi)

    # 6. 计算平均接受率
    avg_accept = accepted_mask.float().mean().item()

    return phi_next, delta_H, delta_inverse_H, delta_phi, delta_p, True, avg_accept


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
    print(f"data.shape:{data.shape}")
    N_bin = data.shape[0]
    lst = []
    for _ in range(bootsrtrap_time):
        idx = torch.randint(0, N_bin, (N_bin,))
        lst.append(data[idx].mean(dim=0))
    return torch.stack(lst)

    for i in range(N_bin):
        idx = torch.randint(0, data.size(0), (1,))
        lst.append(data[idx.item()])
    bootstrap_ensemble = torch.stack(lst)
    # print(f"bootstrap_ensemble shape:{bootstrap_ensemble.shape}")
    return bootstrap_ensemble


def main():
    print(f"Config: L={CONFIG['L']}, Tao={CONFIG['tao']} (No Batch/Channel dims)")

    # 初始化：直接生成 [batchsize, L, L]
    phi = torch.zeros(CONFIG['batch_size'], CONFIG['L'], CONFIG['L'], dtype=DTYPE, device=device)
    delta_H_list = []
    delta_inverse_hamiltonian_list = []
    delta_phi_list = []
    delta_p_list = []
    # 预热 (Thermalization)
    print("Start Thermalization...")
    step = 0
    while step < CONFIG['thermal_steps']:
        phi, delta_H, inverse_hamiltonian, delta_phi, delta_p, success, avg_acc = HMC_step(phi, CONFIG['tao'])

        if not success:
            print(f"Thermal Step {step}: NaN detected! Retrying...")
            if step == 0:
                phi = torch.zeros(CONFIG['batch_size'], CONFIG['L'], CONFIG['L'], dtype=DTYPE, device=device)
            continue

        step += 1
        if step % 1000 == 0:
            print(f"Thermal Step {step}/{CONFIG['thermal_steps']}")

    # 采样 (Sampling)
    print("Start Sampling...")
    ensemble = []
    step = 0
    accept_count = 0
    acc_list = []

    while step < CONFIG['n_samples']:
        phi, delta_H, inverse_hamiltonian, delta_phi, delta_p, success, avg_acc = HMC_step(phi, CONFIG['tao'])

        if not success:
            print(f"Sample Step {step}: NaN detected! Retrying...")
            continue

        accept_count += avg_acc  # 累加平均接受率
        acc_list.append(avg_acc)
        if step % CONFIG['save_steps'] == 0:
            # detach 并存入列表
            ensemble.append(phi.clone().detach().cpu())
            delta_H_list.append(delta_H.detach().cpu())
            delta_inverse_hamiltonian_list.append(inverse_hamiltonian.detach().cpu())
            delta_phi_list.append(delta_phi.detach().cpu())
            delta_p_list.append(delta_p.detach().cpu())


        step += 1
        if step % 1000 == 0:
            print(f"Sampling Step {step}/{CONFIG['n_samples']}, Accept Ratio: {accept_count / step * 100:.2f}%")

    # 处理数据
    # ensemble_tensor shape: [Samples, batchsize, L, L]
    ensemble_tensor = torch.stack(ensemble, dim=0)
    acc_arr = np.array(acc_list)
    acc_mean = acc_arr.mean()
    acc_std = acc_arr.std()
    acc_error = acc_std / np.sqrt(len(acc_arr))

    print(f"平均接受率 (Mean): {acc_mean:.5f}")
    print(f"单次波动 (Std Dev): {acc_std:.5f}")
    print(f"平均值误差 (Std Err): {acc_error:.5f}")

    L = CONFIG['L']
    ensemble_tensor = ensemble_tensor.transpose(0, 1).reshape(-1, L, L)
    print(f"Ensemble shape: {ensemble_tensor.shape}")

    delta_inverse_hamiltonian_tensor = torch.stack(delta_inverse_hamiltonian_list)
    delta_inverse_hamiltonian_tensor = delta_inverse_hamiltonian_tensor.transpose(0, 1).reshape(-1)

    delta_H_tensor = torch.stack(delta_H_list, dim=0)
    delta_H_tensor = delta_H_tensor.transpose(0, 1).reshape(-1)

    delta_phi_list_tensor = torch.stack(delta_phi_list, dim=0)
    delta_phi_list_tensor = delta_phi_list_tensor.transpose(0, 1).reshape(-1, L, L)

    delta_p_list_tensor = torch.stack(delta_p_list, dim=0)
    delta_p_list_tensor = delta_p_list_tensor.transpose(0, 1).reshape(-1, L, L)

    # shape:[N_bin, bin_size]
    binning_delta_H_tensor = binning(delta_H_tensor)
    # mean in bin
    binning_delta_H_tensor = binning_delta_H_tensor.mean(dim=1)
    bootstrap_delta_H_tensor = bootstrap(binning_delta_H_tensor, CONFIG['bootstrap_time'])
    delta_H_error = bootstrap_delta_H_tensor.std(dim=0)
    print(f'expect of delta_H:{torch.mean(delta_H_tensor).item():.2e}')
    print(f"{torch.mean(binning_delta_H_tensor):.5f}({int(delta_H_error * 1e5):02d})")
    delta_H_exp_tensor = torch.exp(-delta_H_tensor)
    binning_delta_H_exp_tensor = binning(delta_H_exp_tensor)
    binning_delta_H_exp_tensor = binning_delta_H_exp_tensor.mean(dim=0)
    bootstrap_delta_H_exp_tensor = bootstrap(binning_delta_H_exp_tensor, CONFIG['bootstrap_time'])
    delta_H_exp_error = bootstrap_delta_H_exp_tensor.std(dim=0)

    print(f'expect of delta_H_exp:{torch.mean(binning_delta_H_exp_tensor).item():.2e}')
    print(f'{torch.mean(binning_delta_H_exp_tensor):.2e}({int(delta_H_exp_error * 1e5):02d})')
    print(f'delta_H_exp_error:{delta_H_exp_error:.2e}')

    # print(f"ensemble_mean:{ensemble_tensor.mean().item():.2e}")
    # print(f'delta_inverse_hamiltonian_list:{delta_inverse_hamiltonian_list}')
    # delta_inverse_hamiltonian_tensor = torch.cat(delta_inverse_hamiltonian_list)
    # y = delta_inverse_hamiltonian_tensor  # 你的 1D torch tensor
    y = np.array([x.item() for x in delta_inverse_hamiltonian_tensor.cpu().numpy()])
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
    bootstrap_phi_to_1 = bootstrap(binning_ensemble.mean(dim=1), CONFIG['bootstrap_time']).mean(dim=(1,2))
    print(f"bootstrap_phi_to_1 shape: {bootstrap_phi_to_1.shape}")
    phi_to_1_error = bootstrap_phi_to_1.std(dim=0)
    print(f"phi_to_1_error shape: {phi_to_1_error.shape}")
    phi_to_2 = (binning_ensemble * binning_ensemble).mean()
    bootstrap_phi_to_2 = bootstrap((binning_ensemble**2).mean(dim=1), CONFIG['bootstrap_time']).mean(dim=(1,2))
    phi_to_2_error = bootstrap_phi_to_2.std(dim=0)
    phi_to_3 = (binning_ensemble **3).mean()
    bootstrap_phi_to_3 = bootstrap((binning_ensemble**3).mean(dim=1), CONFIG['bootstrap_time']).mean(dim=(1,2))
    phi_to_3_error = bootstrap_phi_to_3.std(dim=0)
    phi_to_4 = (binning_ensemble ** 4).mean()
    bootstrap_phi_to_4 = bootstrap((binning_ensemble**4).mean(dim=1), CONFIG['bootstrap_time']).mean(dim=(1,2))
    phi_to_4_error = bootstrap_phi_to_4.std(dim=0)
    phi_to_5 = (binning_ensemble ** 5).mean()
    bootstrap_phi_to_5 = bootstrap((binning_ensemble**5).mean(dim=1), CONFIG['bootstrap_time']).mean(dim=(1,2))
    phi_to_5_error = bootstrap_phi_to_5.std(dim=0)
    print("delta_H: {:.2e}".format(delta_H_tensor.mean().item()))
    print("expectation of exp delta_h: {:.2e}".format(torch.exp(-delta_H_tensor).mean().item()))
    print(f"phi_to_1:{phi_to_1:.2e}")
    print(f"{phi_to_1:.6f}({phi_to_1_error:.6f})")
    print(f"phi_to_2:{phi_to_2:.2e}")
    print(f"{phi_to_2:.6f}({phi_to_2_error:.6f})")
    print(f"phi_to_3:{phi_to_3:.2e}")
    print(f"{phi_to_3:.6f}({phi_to_3_error:.6f})")
    print(f"phi_to_4:{phi_to_4:.2e}")
    print(f"{phi_to_4:.6f}({phi_to_4_error:.6f})")
    print(f"phi_to_5:{phi_to_5:.2e}")
    print(f"{phi_to_5:.6f}({phi_to_5_error:.6f})")
    try:
        # shape: N_bin, t
        G_t_inside_bin = calculate_G_t_list_inside_bin(binning_ensemble)
        print(f"G_t_inside_bin's shape:{G_t_inside_bin.shape}")
        # shape=(bootsraptime, N_bin, time)

        # for i in range(CONFIG['bootstrap_time']):
        #     bootstrap_tensor.append(bootstrap(G_t_inside_bin))
        # shape=(bootsraptime, time)

        bootstrap_ensemble_tensor = bootstrap(G_t_inside_bin, CONFIG['bootstrap_time'])
        # green function
        # shape=(bootsraptime, time)
        # bootstrap_tensor = torch.stack(bootstrap_tensor).mean(dim=1)
        print(f"bootstrap_tensor's shape:{bootstrap_ensemble_tensor.shape}")
        G_error_bar = bootstrap_ensemble_tensor.std(dim=0)
        y_err_g = G_error_bar.numpy()
        G_t = G_t_inside_bin.mean(dim=0)
        print("G_t values:")
        for t_val in G_t:
            print(f"{t_val.item():.16E}")

        print("y_err_g:")
        for t_val in y_err_g:
            print(f"{t_val.item():.16E}")

        # effective mass
        # shape=(time)
        effective_mass = get_effective_mass(G_t)
        effective_mass = effective_mass[1:-1]
        bootstrap_effective_mass = get_effective_mass(bootstrap_ensemble_tensor)
        bootstrap_effective_mass = bootstrap_effective_mass[:, 1:-1]
        e_m_error_bar = bootstrap_effective_mass.std(dim=0)
        y_err_m = e_m_error_bar.numpy()
        # 1. 设置画布
        # 两个图都是 1.4:1 的宽图，并排显示，建议把画布宽度设大一点，比如 (10, 4) 或 (12, 5)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # --- 左图：Green Function G(t) ---

        # 设置宽高比 W:H = 1.4:1 -> H/W = 1/1.4
        axs[0].set_box_aspect(1 / 1.6875)

        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].errorbar(
            np.arange(len(G_t.numpy())),
            G_t.numpy(),
            yerr=y_err_g,
            fmt='-o',  # 格式: '-'连线, 'o'点
            color='blue',  # 数据颜色
            ecolor='purple',  # 误差棒颜色
            capsize=4,  # 误差棒帽子宽度
            elinewidth=1.5,  # 误差棒线宽
            label='Lattice Data'
        )
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Time Separation (t)')
        axs[0].set_ylabel('G(t) (Connected)')
        axs[0].set_title(f'2-point function (L={CONFIG["L"]}, tao={CONFIG["tao"]})')
        axs[0].grid(True, which="both", ls="--", alpha=0.5)  # 加了 alpha 让网格淡一点，不抢眼
        axs[0].legend()  # 显示图例

        # --- 右图：Effective Mass ---

        # 设置宽高比 W:H = 1.4:1
        axs[1].set_box_aspect(1 / 1.6875)

        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].errorbar(
            np.arange(len(e_m_error_bar.numpy())),  # 假设这是对应的时间切片长度
            effective_mass.numpy(),
            yerr=y_err_m,
            fmt='-o',
            color='blue',
            ecolor='purple',
            capsize=4,
            elinewidth=1.5,
            label='Lattice Data'
        )
        axs[1].set_xlabel('Time Separation (t)')
        axs[1].set_ylabel('Effective Mass')
        axs[1].set_title(f'Effective Mass (L={CONFIG["L"]}, tao={CONFIG["tao"]})')
        # axs[1].grid(True, which="both", ls="--") # 你之前注释掉了，我也保持注释状态
        axs[1].legend()

        # 3. 布局调整与显示
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
