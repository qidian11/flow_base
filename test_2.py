import torch
import matplotlib.pyplot as plt

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'tao': 1.2,
    'leap_frog_step': 10,
    'save_steps': 10,
    'thermal_steps': 1000,
    'n_samples': 3000,
    # 'batch_size': 1,  <-- 已移除，代码现在纯粹是单链 2D 张量操作
}

# 确保使用双精度
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
            return phi_new, p_new

        if i != CONFIG['leap_frog_step'] - 1:
            p_new = p_new + epsilon * get_force(phi_new)

    # Final half step for momentum
    p_new = p_new + (epsilon / 2) * get_force(phi_new)

    return phi_new, p_new


def HMC_step_single_chain(phi, tao):
    """
    输入 phi shape: [L, L]
    """
    # 1. 重新采样动量，形状直接为 [L, L]
    p = torch.randn(CONFIG['L'], CONFIG['L'], dtype=DTYPE)

    hamiltonian_old = calculate_hamiltonian(phi, p)

    # 2. Leapfrog 演化
    phi_new, p_new = leap_frog(phi, p, tao)

    # --- NaN 检测 ---
    if torch.isnan(phi_new).any() or torch.isinf(phi_new).any():
        return phi, False, False

    hamiltonian_new = calculate_hamiltonian(phi_new, p_new)

    # 计算能量差
    delta_H = hamiltonian_new - hamiltonian_old

    if torch.isnan(delta_H) or torch.isinf(delta_H):
        return phi, False, False

    # 3. Metropolis 接受/拒绝
    prob = torch.exp(-delta_H).item()
    rand_num = torch.rand(1).item()

    if rand_num < prob:
        return phi_new, True, True
    else:
        return phi, True, False


def calculate_G_t_list(ensemble):
    """
    计算 G(t)。
    ensemble shape: [N_samples, L, L]
    (已经去除了 Batch 和 Channel)
    """
    N, L, _ = ensemble.shape
    G_t = []

    # 假设 dim 1 是时间 T，dim 2 是空间 X
    # phi_mean_spatial shape: [N, T]
    phi_mean_spatial = ensemble.mean(dim=2)
    phi_bar = phi_mean_spatial.mean()  # 全局平均值

    for t in range(L):
        corrs = []
        for t0 in range(L):
            t_next = (t0 + t) % L
            val = (phi_mean_spatial[:, t_next] - phi_bar) * (phi_mean_spatial[:, t0] - phi_bar)
            corrs.append(val.mean())
        G_t.append(torch.tensor(corrs).mean())

    return torch.stack(G_t)


def main():
    print(f"Config: L={CONFIG['L']}, Tao={CONFIG['tao']} (No Batch/Channel dims)")

    # 初始化：直接生成 [L, L]
    phi = torch.randn(CONFIG['L'], CONFIG['L'], dtype=DTYPE)

    # 预热 (Thermalization)
    print("Start Thermalization...")
    step = 0
    while step < CONFIG['thermal_steps']:
        phi, success, accepted = HMC_step_single_chain(phi, CONFIG['tao'])

        if not success:
            print(f"Thermal Step {step}: NaN detected! Retrying...")
            if step == 0:
                phi = torch.randn(CONFIG['L'], CONFIG['L'], dtype=DTYPE)
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
        phi, success, accepted = HMC_step_single_chain(phi, CONFIG['tao'])

        if not success:
            print(f"Sample Step {step}: NaN detected! Retrying...")
            continue

        if accepted:
            accept_count += 1

        if step % CONFIG['save_steps'] == 0:
            # detach 并存入列表
            ensemble.append(phi.clone().detach())

        step += 1
        if step % 100 == 0:
            print(f"Sampling Step {step}/{CONFIG['n_samples']}, Accept Ratio: {accept_count / step * 100:.2f}%")

    # 处理数据
    # ensemble_tensor shape: [Samples, L, L]
    ensemble_tensor = torch.stack(ensemble, dim=0)
    print(f"Ensemble shape: {ensemble_tensor.shape}")

    try:
        G_t = calculate_G_t_list(ensemble_tensor)

        plt.figure(figsize=(8, 6))
        plt.plot(G_t.numpy(), marker='o', linestyle='-')
        plt.yscale('log')
        plt.xlabel('Time Separation (t)')
        plt.ylabel('G(t) (Connected)')
        plt.title(f'2-point function (L={CONFIG["L"]}, tao={CONFIG["tao"]})')
        plt.grid(True, which="both", ls="--")
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()