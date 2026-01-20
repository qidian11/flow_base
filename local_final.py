import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- 设备与配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'delta': 0.5,
    'save_steps': 10,
    'thermal_steps': 2000,
    'n_samples': 20000,
    'bin_size': 100,
    'bootstrap_time': 1000,
    'batch_size': 128,
}


# --- 核心物理算法 ---
def get_neighbor_sum(phi):
    up = torch.roll(phi, shifts=1, dims=1)
    down = torch.roll(phi, shifts=-1, dims=1)
    left = torch.roll(phi, shifts=1, dims=2)
    right = torch.roll(phi, shifts=-1, dims=2)
    return up + down + left + right


def checkerboard_metropolis_batch(phi, mask_red, mask_black):
    L, m2, lam, delta = CONFIG['L'], CONFIG['m2'], CONFIG['lam'], CONFIG['delta']
    term1 = (4 + m2)

    total_dS = torch.zeros(phi.shape[0], device=DEVICE)
    total_acc = 0.0

    for mask in [mask_red, mask_black]:
        neighbor_sum = get_neighbor_sum(phi)
        change = (torch.rand_like(phi) - 0.5) * 2 * delta
        phi_proposal = phi + change

        # Delta S 计算 (基于 local_test_7 的 2.0 修正)
        dS_grid = (term1 * (phi_proposal ** 2 - phi ** 2) +
                   lam * (phi_proposal ** 4 - phi ** 4) -
                   (phi_proposal - phi) * neighbor_sum)

        accept_condition = (dS_grid < 0) | (torch.exp(-dS_grid) > torch.rand_like(phi))
        update_mask = mask & accept_condition

        # 记录每条链的 Action 变化
        total_dS += (dS_grid * mask.float()).sum(dim=(1, 2))
        phi = torch.where(update_mask, phi_proposal, phi)

        # 统计接受率：各相均值之和 = 总接受比例
        total_acc += update_mask.float().mean().item()

    return phi, total_dS, total_acc


# --- 统计分析 ---
def binning(data):
    N_bin = data.shape[0] // CONFIG['bin_size']
    return data[:N_bin * CONFIG['bin_size']].view(N_bin, CONFIG['bin_size'], *data.shape[1:])


def bootstrap(data, count):
    N_bin = data.shape[0]
    indices = torch.randint(0, N_bin, (count, N_bin))
    return data[indices].mean(dim=1)


def calculate_G_t_bins(binning_ens):
    L = CONFIG['L']
    expected_phi = binning_ens.mean(dim=(1, 2, 3))  # [N_bin]
    G_t_list = []
    for t in range(L):
        G_t_l_sum = []
        for l in range(L):
            shifted = torch.roll(binning_ens, shifts=(t, l), dims=(2, 3))
            corr = (binning_ens * shifted).mean(dim=(2, 3)).mean(dim=1)
            G_t_l_sum.append(corr - expected_phi *(shifted.mean(dim=(1, 2, 3))))
        G_t_list.append(torch.stack(G_t_l_sum).mean(dim=0))
    return torch.stack(G_t_list, dim=1)


def main():
    print(f"Running on {DEVICE}...")
    phi = torch.ones(CONFIG['batch_size'], CONFIG['L'], CONFIG['L'], device=DEVICE) * 0.6
    coords = torch.arange(CONFIG['L'], device=DEVICE)
    i, j = torch.meshgrid(coords, coords, indexing='ij')
    mask_red = ((i + j) % 2 == 0).unsqueeze(0).expand(CONFIG['batch_size'], -1, -1)
    mask_black = ((i + j) % 2 == 1).unsqueeze(0).expand(CONFIG['batch_size'], -1, -1)

    # 1. Thermalization
    for _ in range(CONFIG['thermal_steps']):
        phi, _, _ = checkerboard_metropolis_batch(phi, mask_red, mask_black)

    # 2. Sampling
    ensemble, dS_list, acc_sum = [], [], 0.0
    acc_list = []
    steps = CONFIG['n_samples']
    for s in range(steps):
        phi, dS, acc = checkerboard_metropolis_batch(phi, mask_red, mask_black)
        acc_sum += acc
        acc_list.append(acc)
        if s % CONFIG['save_steps'] == 0:
            ensemble.append(phi.clone().cpu())
            dS_list.append(dS.clone().cpu())
        if s % 1000 == 0:
            print(f"Step {s}/{steps}, Acc: {acc_sum / (s + 1) * 100:.2f}%")

    # 3. Process Data
    ens_tensor = torch.stack(ensemble).transpose(0, 1).reshape(-1, CONFIG['L'], CONFIG['L'])
    dS_tensor = torch.stack(dS_list).transpose(0, 1).reshape(-1)
    bin_ens = binning(ens_tensor)

    # 4. Output Observables (仿照 test_4)
    print("\n--- Statistics ---")
    for n in range(1, 6):
        obs_bin = (bin_ens ** n).mean(dim=(1, 2, 3))
        boot = bootstrap(obs_bin, CONFIG['bootstrap_time'])
        print(f"phi^{n}: {boot.mean():.6f}({boot.std():.6f})")

    # 5. Physics Check
    exp_dS = torch.exp(-dS_tensor)
    print(f"exp(-delta_S): {exp_dS.mean():.5f}")

    # 6. G(t) & Mass Calculation
    G_t_bins = calculate_G_t_bins(bin_ens)
    G_boot = bootstrap(G_t_bins, CONFIG['bootstrap_time'])
    G_mean, G_err = G_boot.mean(dim=0), G_boot.std(dim=0)

    m_eff_boot = torch.acosh(((torch.roll(G_boot, -1, -1) + torch.roll(G_boot, 1, -1)) / (2 * G_boot)).clamp(min=1.0))
    m_mean, m_err = m_eff_boot.mean(dim=0)[1:-1], m_eff_boot.std(dim=0)[1:-1]

    # 7. 绘图优化 (核心修改点)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    t_axis = np.arange(CONFIG['L'])
    m_axis = t_axis[1:-1]

    # --- 左图: G(t) ---
    axs[0].errorbar(t_axis, G_mean.numpy(), yerr=G_err.numpy(), fmt='-o', color='blue', capsize=5, label='$G(t)$')
    axs[0].set_yscale('log')
    axs[0].set_title(f"Connected Correlation $G(t)$ (L={CONFIG['L']})", fontsize=12)
    axs[0].set_xlabel('Time $t$', fontsize=10)
    axs[0].set_ylabel('$G(t)$', fontsize=10)
    axs[0].set_xticks(t_axis)
    # 强制显示科学计数法标签，不仅仅是 10^-2
    axs[0].yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    axs[0].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * .1, numticks=10))
    axs[0].grid(True, which="both", ls="--", alpha=0.5)

    # --- 右图: Effective Mass ---
    axs[1].errorbar(m_axis, m_mean.numpy(), yerr=m_err.numpy(), fmt='-s', color='red', capsize=5, label='$m_{eff}$')
    axs[1].set_title("Effective Mass $m_{eff}(t)$", fontsize=12)
    axs[1].set_xlabel('Time $t$', fontsize=10)
    axs[1].set_ylabel('$m_{eff}$', fontsize=10)
    axs[1].set_xticks(m_axis)
    # 动态缩放：根据数据范围设置，增加上下各 15% 的边距
    y_min, y_max = m_mean.min(), m_mean.max()
    pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.1
    axs[1].set_ylim(y_min - pad, y_max + pad)
    axs[1].grid(True, which="major", ls="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()