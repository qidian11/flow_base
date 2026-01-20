import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator


# --- Device Configuration ---
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


DEVICE = get_device(prefer="auto")
print(f"🔥 Device: {DEVICE}")

DTYPE = torch.float32

CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'delta': 0.5,
    'save_steps': 10,
    'thermal_steps': 2000,
    'n_samples': 40000,  # 稍微增加采样数以匹配 HMC 的统计质量
    'bin_size': 100,
    'bootstrap_time': 1000,
    'batch_size': 128,
}


# --- Core Physics Algorithms (Metropolis) ---

def get_neighbor_sum(phi):
    up = torch.roll(phi, shifts=-1, dims=1)
    down = torch.roll(phi, shifts=1, dims=1)
    left = torch.roll(phi, shifts=-1, dims=2)
    right = torch.roll(phi, shifts=1, dims=2)
    return up + down + left + right


def checkerboard_metropolis_batch(phi, mask_red, mask_black):
    # 物理修正：严格对齐 test_4 的 Action 定义
    # S = sum( (4+m2)phi^2 + lam*phi^4 - phi * neighbor_sum )
    L, m2, lam, delta = CONFIG['L'], CONFIG['m2'], CONFIG['lam'], CONFIG['delta']
    term1 = (4 + m2)

    total_dS = torch.zeros(phi.shape[0], device=DEVICE)
    total_acc = 0.0

    for mask in [mask_red, mask_black]:
        neighbor_sum = get_neighbor_sum(phi)

        # Proposal
        change = (torch.rand_like(phi) - 0.5) * 2 * delta
        phi_proposal = phi + change

        # --- 核心修正点 ---
        # 邻居项必须乘以 2，因为 Action 是全局求和，改变 phi_x 既改变了 x 的项，也改变了邻居 y 的项。
        # test_4 force: -(8*phi - 2*neighbors ...) -> 2 * neighbors
        # Metropolis dS 对应: -2 * d_phi * neighbors
        dS_grid = (term1 * (phi_proposal ** 2 - phi ** 2) +
                   lam * (phi_proposal ** 4 - phi ** 4) -
                   2 * (phi_proposal - phi) * neighbor_sum)  # <--- 必须乘 2 !!!

        # Metropolis Acceptance
        rand_val = torch.rand_like(phi)
        accept_condition = (dS_grid < 0) | (torch.exp(-dS_grid) > rand_val)

        update_mask = mask & accept_condition

        # 仅累加被接受的 steps 的 dS (你指出的修正)
        current_dS_change = (dS_grid * update_mask.float()).sum(dim=(1, 2))
        total_dS += current_dS_change

        # Update field
        phi = torch.where(update_mask, phi_proposal, phi)

        # Track acceptance rate
        total_acc += update_mask.float().mean().item()

    return phi, total_dS, total_acc


# --- Statistical Analysis (Strictly following test_4) ---

def binning(data):
    N_bin = data.shape[0] // CONFIG['bin_size']
    data = data[:N_bin * CONFIG['bin_size']]
    data = data.view(N_bin, CONFIG['bin_size'], *data.shape[1:])
    return data


def bootstrap(data, bootstrap_time):
    N_bin = data.shape[0]
    lst = []
    for _ in range(bootstrap_time):
        idx = torch.randint(0, N_bin, (N_bin,), device=data.device)
        lst.append(data[idx].mean(dim=0))
    return torch.stack(lst)


def get_expected_phi(binning_ensemble):
    # 严格匹配 test_4: 先对 (0,1) 求平均得到 [L,L]，再对 (0,1) 求平均得到 scalar
    # 注意 test_4 里的 binning_ensemble 已经是 [N_bin, bin_size, L, L]
    m = binning_ensemble.mean(dim=(0, 1))
    return m.mean()


def get_correlation_for_single_configuration(binning_ensemble, time_shift, space_shift):
    ensemble_shifted = torch.roll(binning_ensemble, shifts=(time_shift, space_shift), dims=(2, 3))
    corr = ensemble_shifted * binning_ensemble
    return corr.mean(dim=(2, 3))  # [N_bin, bin_size]


def calculate_G_t_list_inside_bin(binning_ensemble):
    # 严格匹配 test_4 逻辑
    G_t_list = []
    expected_phi = get_expected_phi(binning_ensemble)

    for t in range(CONFIG['L']):
        G_t_l_list = []
        for l in range(CONFIG['L']):
            conn_2_point = get_correlation_for_single_configuration(binning_ensemble, t, l)
            conn = conn_2_point - expected_phi * expected_phi
            G_t_l_list.append(conn)

        G_t_mean_tensor = torch.stack(G_t_l_list, dim=1)
        G_t_val = G_t_mean_tensor.mean(dim=(1, 2))  # Mean over Space and Bin_size
        G_t_list.append(G_t_val)

    return torch.stack(G_t_list, dim=1)


def get_effective_mass(G_t):
    G_plus = torch.roll(G_t, shifts=-1, dims=-1)
    G_minus = torch.roll(G_t, shifts=1, dims=-1)
    cosh_m = (G_plus + G_minus) / (2 * G_t + 1e-10)
    cosh_m = torch.clamp(cosh_m, min=1.0)
    m_eff = torch.acosh(cosh_m)
    return m_eff


# --- Main ---
def main():
    print(f"Running on {DEVICE}...")

    phi = torch.ones(CONFIG['batch_size'], CONFIG['L'], CONFIG['L'], device=DEVICE) * 0.6

    coords = torch.arange(CONFIG['L'], device=DEVICE)
    i, j = torch.meshgrid(coords, coords, indexing='ij')
    mask_red = ((i + j) % 2 == 0).unsqueeze(0).expand(CONFIG['batch_size'], -1, -1)
    mask_black = ((i + j) % 2 == 1).unsqueeze(0).expand(CONFIG['batch_size'], -1, -1)

    # 1. Thermalization
    print(f"Thermalization ({CONFIG['thermal_steps']} steps)...")
    for _ in range(CONFIG['thermal_steps']):
        phi, _, _ = checkerboard_metropolis_batch(phi, mask_red, mask_black)

    # 2. Sampling
    print(f"Sampling ({CONFIG['n_samples']} steps)...")
    ensemble = []
    dS_list = []
    acc_sum = 0.0

    steps = CONFIG['n_samples']
    for s in range(steps):
        phi, dS, acc = checkerboard_metropolis_batch(phi, mask_red, mask_black)
        acc_sum += acc

        if s % CONFIG['save_steps'] == 0:
            ensemble.append(phi.clone().cpu())
            dS_list.append(dS.clone().cpu())

        if s % 5000 == 0:
            print(f"Step {s}/{steps}, Acc: {acc_sum / (s + 1) * 100:.2f}%")

    # 3. Data Processing
    # 这里的 reshape 逻辑必须极其小心，确保时序连续性
    # test_4 是 [Samples, Batch, L, L] -> transpose(0,1) -> [Batch, Samples, L, L] -> flatten
    # 这意味着把每个 Batch 作为一个独立的长链连接起来
    ensemble_tensor = torch.stack(ensemble).transpose(0, 1).reshape(-1, CONFIG['L'], CONFIG['L'])
    dS_tensor = torch.stack(dS_list).transpose(0, 1).reshape(-1)

    print(f"Ensemble shape: {ensemble_tensor.shape}")

    binning_ensemble = binning(ensemble_tensor)

    print("\n--- Observables ---")
    # Observables
    for n in range(1, 6):
        obs = binning_ensemble ** n
        boot_res = bootstrap(obs.mean(dim=1), CONFIG['bootstrap_time']).mean(dim=(1, 2))
        print(f"phi^{n}: {obs.mean().item():.6f}({boot_res.std().item():.6f})")

    # Action Check
    exp_dS_tensor = torch.exp(-dS_tensor)
    print(f"exp(-delta_S) mean: {exp_dS_tensor.mean().item():.5f} (Should be close to 1.0)")

    # 4. G(t) & Effective Mass
    G_t_inside_bin = calculate_G_t_list_inside_bin(binning_ensemble)

    bootstrap_G = bootstrap(G_t_inside_bin, CONFIG['bootstrap_time'])
    G_mean = bootstrap_G.mean(dim=0)
    G_err = bootstrap_G.std(dim=0)

    print("\nG_t values:")
    for t_val in G_mean:
        print(f"{t_val.item():.5E}")

    m_eff_boot = get_effective_mass(bootstrap_G)
    # slice [1:-1]
    m_eff_boot = m_eff_boot[:, 1:-1]
    m_mean = m_eff_boot.mean(dim=0)
    m_err = m_eff_boot.std(dim=0)

    # 5. Plotting
    t_axis = np.arange(CONFIG['L'])
    m_axis = t_axis[1:-1]

    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Left: G(t)
        axs[0].set_box_aspect(1 / 1.6875)
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].errorbar(t_axis, G_mean.numpy(), yerr=G_err.numpy(), fmt='-o',
                        color='blue', ecolor='purple', capsize=4, elinewidth=1.5, label='Local Metropolis')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('G(t)')
        axs[0].set_title(f'2-point Function (L={CONFIG["L"]})')
        axs[0].grid(True, which="both", ls="--", alpha=0.5)
        axs[0].legend()

        # Right: Effective Mass
        axs[1].set_box_aspect(1 / 1.6875)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].errorbar(m_axis, m_mean.numpy(), yerr=m_err.numpy(), fmt='-s',
                        color='red', ecolor='purple', capsize=4, elinewidth=1.5, label='Effective Mass')
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('Mass')
        axs[1].set_title(f'Effective Mass')

        # Auto-scale Y with padding
        y_min, y_max = m_mean.min().item(), m_mean.max().item()
        pad = (y_max - y_min) * 0.5 if y_max != y_min else 0.1
        axs[1].set_ylim(y_min - pad, y_max + pad)
        axs[1].grid(True, ls="--", alpha=0.7)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Plotting error: {e}")


if __name__ == "__main__":
    main()