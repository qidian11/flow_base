import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# --- 设备配置 ---
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
print(f"🔥 当前测试设备: {DEVICE}")

# 1. 核心修改：设置 Local 为双精度
DTYPE = torch.float64

# 3. 仿照 HMC，构建多晶格尺寸的配置列表 (L=8 到 L=14)
# 物理参数 m2, lam 严格对齐 HMC 晶格对应的临界点参数
CONFIGS = [
    {
        'L': 6,
        'm2':-4,
        'lam':6.975,
        'delta': 0.5,
        'save_steps': 10,
        'thermal_steps': 2000,
        'n_samples': 100000,
        'bin_size': 100,
        'bootstrap_time': 1000,
        'batch_size': 128,
    },
    {
        'L': 8,
        'm2': -4.0,
        'lam': 6.008,
        'delta': 0.5,
        'save_steps': 10,
        'thermal_steps': 2000,
        'n_samples': 100000,
        'bin_size': 100,
        'bootstrap_time': 1000,
        'batch_size': 128,
    },
    {
        'L': 10,
        'm2': -4.0,
        'lam': 5.550,
        'delta': 0.5,
        'save_steps': 10,
        'thermal_steps': 2000,
        'n_samples': 100000,
        'bin_size': 100,
        'bootstrap_time': 1000,
        'batch_size': 128,
    },
    {
        'L': 12,
        'm2': -4.0,
        'lam': 5.276,
        'delta': 0.5,
        'save_steps': 10,
        'thermal_steps': 2000,
        'n_samples': 100000,
        'bin_size': 100,
        'bootstrap_time': 1000,
        'batch_size': 128,
    },
    {
        'L': 14,
        'm2': -4.0,
        'lam': 5.113,
        'delta': 0.5,
        'save_steps': 10,
        'thermal_steps': 2000,
        'n_samples': 100000,
        'bin_size': 100,
        'bootstrap_time': 1000,
        'batch_size': 128,
    }
]


# --- 核心 Metropolis 算法 ---

def get_neighbor_sum(phi):
    up = torch.roll(phi, shifts=-1, dims=1)
    down = torch.roll(phi, shifts=1, dims=1)
    left = torch.roll(phi, shifts=-1, dims=2)
    right = torch.roll(phi, shifts=1, dims=2)
    return up + down + left + right


def checkerboard_metropolis_batch(phi, mask_red, mask_black, config):
    # S = sum( (4+m2)phi^2 + lam*phi^4 - phi * neighbor_sum )
    m2, lam, delta = config['m2'], config['lam'], config['delta']
    term1 = (4 + m2)
    total_acc = 0.0

    for mask in [mask_red, mask_black]:
        neighbor_sum = get_neighbor_sum(phi)

        # Proposal (自动继承双精度)
        change = (torch.rand_like(phi) - 0.5) * 2 * delta
        phi_proposal = phi + change

        # 计算单格点作用量变化（考虑全格点求和，邻居项包含因子 2）
        dS_grid = (term1 * (phi_proposal ** 2 - phi ** 2) +
                   lam * (phi_proposal ** 4 - phi ** 4) -
                   2 * (phi_proposal - phi) * neighbor_sum)

        # Metropolis 接受判定
        rand_val = torch.rand_like(phi)
        accept_condition = (dS_grid < 0) | (torch.exp(-dS_grid) > rand_val)

        update_mask = mask & accept_condition

        # 更新场位形
        phi = torch.where(update_mask, phi_proposal, phi)

        # 记录接受率
        total_acc += update_mask.float().mean().item()

    # 2. 彻底去掉了无用的 total_dS 追踪，返回值更干净
    return phi, total_acc


# --- 统计分析模块 ---

def binning(data, config):
    N_bin = data.shape[0] // config['bin_size']
    data = data[:N_bin * config['bin_size']]
    data = data.view(N_bin, config['bin_size'], *data.shape[1:])
    return data


def bootstrap(data, bootstrap_time):
    N_bin = data.shape[0]
    lst = []
    for _ in range(bootstrap_time):
        idx = torch.randint(0, N_bin, (N_bin,), device=data.device)
        lst.append(data[idx].mean(dim=0))
    return torch.stack(lst)


def get_expected_phi(binning_ensemble):
    m = binning_ensemble.mean(dim=(0, 1))
    return m.mean()


def get_correlation_for_single_configuration(binning_ensemble, time_shift, space_shift):
    ensemble_shifted = torch.roll(binning_ensemble, shifts=(time_shift, space_shift), dims=(2, 3))
    corr = ensemble_shifted * binning_ensemble
    return corr.mean(dim=(2, 3))


def calculate_G_t_list_inside_bin(binning_ensemble, config):
    G_t_list = []
    expected_phi = get_expected_phi(binning_ensemble)

    for t in range(config['L']):
        G_t_l_list = []
        for l in range(config['L']):
            conn_2_point = get_correlation_for_single_configuration(binning_ensemble, t, l)
            conn = conn_2_point - expected_phi * expected_phi
            G_t_l_list.append(conn)

        G_t_mean_tensor = torch.stack(G_t_l_list, dim=1)
        G_t_val = G_t_mean_tensor.mean(dim=(1, 2))  # 对空间体积及 bin_size 求平均
        G_t_list.append(G_t_val)

    return torch.stack(G_t_list, dim=1)


def get_effective_mass(G_t):
    G_plus = torch.roll(G_t, shifts=-1, dims=-1)
    G_minus = torch.roll(G_t, shifts=1, dims=-1)
    cosh_m = (G_plus + G_minus) / (2 * G_t + 1e-10)
    cosh_m = torch.clamp(cosh_m, min=1.0)
    m_eff = torch.acosh(cosh_m)
    return m_eff


# --- 主模拟循环 ---
def main(config):
    print(f"\n" + "=" * 50)
    print(f"开始运行配置: L={config['L']}, m2={config['m2']}, lam={config['lam']}")
    print(f"参数信息: Batch Size={config['batch_size']}, Total Samples={config['n_samples']}")
    print("=" * 50)

    # 初始化场位形（采用指定的双精度 DTYPE）
    phi = torch.ones(config['batch_size'], config['L'], config['L'], dtype=DTYPE, device=DEVICE) * 0.6

    # 棋盘格掩码初始化
    coords = torch.arange(config['L'], device=DEVICE)
    i, j = torch.meshgrid(coords, coords, indexing='ij')
    mask_red = ((i + j) % 2 == 0).unsqueeze(0).expand(config['batch_size'], -1, -1)
    mask_black = ((i + j) % 2 == 1).unsqueeze(0).expand(config['batch_size'], -1, -1)

    # 1. Thermalization
    print(f"正在进行预热 ({config['thermal_steps']} 步)...")
    for _ in range(config['thermal_steps']):
        phi, _ = checkerboard_metropolis_batch(phi, mask_red, mask_black, config)

    # 2. Sampling
    print(f"正在进行采样 ({config['n_samples']} 步)...")
    ensemble = []
    acc_list = []
    acc_sum = 0.0

    steps = config['n_samples']
    for s in range(steps):
        phi, acc = checkerboard_metropolis_batch(phi, mask_red, mask_black, config)
        acc_sum += acc
        acc_list.append(acc)  # 记录完整的接受率历史

        if s % config['save_steps'] == 0:
            ensemble.append(phi.clone().cpu())

        if s % 5000 == 0:
            print(f"Step {s}/{steps}, 实时平均接受率: {acc_sum / (s + 1) * 100:.2f}%")

    # 3. 数据重组与并行链展开
    ensemble_tensor = torch.stack(ensemble).transpose(0, 1).reshape(-1, config['L'], config['L'])
    print(f"最终系综形状 (Ensemble shape): {ensemble_tensor.shape}")

    # --- 仿照 HMC：保存所有构型与接受率 ---
    print("正在以标准格式 [N, 1, L, L] 导出并压缩保存位形...")

    # 增加通道轴以对齐神经网络输入标准
    standard_ensemble = ensemble_tensor.unsqueeze(1).cpu().numpy()

    # 同样每隔 100 个存储点抽取 1 个进行稀疏化
    # 不稀疏
    saved_ensemble = standard_ensemble[:1000000]

    dtype_str = "double" if DTYPE == torch.float64 else "single"
    save_filename = f"Local_configs_L{config['L']}_N{ensemble_tensor.shape[0]}_DTYPE_{dtype_str}.npz"

    np.savez_compressed(
        save_filename,
        configs=saved_ensemble,
        accept_history=np.array(acc_list)
    )
    print(f"✅ 已保存标准化构型到: {save_filename}")
    print(f"   保存的物理形状 (Saved Shape): {saved_ensemble.shape}")

    # 4. 物理观测盘统计
    binning_ensemble = binning(ensemble_tensor, config)
    print("\n--- 物理观测项 (Observables) ---")
    for n in range(1, 6):
        obs = binning_ensemble ** n
        boot_res = bootstrap(obs.mean(dim=1), config['bootstrap_time']).mean(dim=(1, 2))
        print(f"phi^{n}: {obs.mean().item():.6f}({boot_res.std().item():.6f})")

    # 2. 彻底移除了原先在此处的错误的 action 检验逻辑

    # 5. 计算 G(t) 与 有效质量 (Effective Mass)
    G_t_inside_bin = calculate_G_t_list_inside_bin(binning_ensemble, config)

    bootstrap_G = bootstrap(G_t_inside_bin, config['bootstrap_time'])
    G_mean = bootstrap_G.mean(dim=0)
    G_err = bootstrap_G.std(dim=0)

    print("\nG_t 关联函数中心值:")
    for t_val in G_mean:
        print(f"{t_val.item():.5E}")

    m_eff_boot = get_effective_mass(bootstrap_G)
    m_eff_boot = m_eff_boot[:, 1:-1]
    m_mean = m_eff_boot.mean(dim=0)
    m_err = m_eff_boot.std(dim=0)

    # 6. 绘图输出
    t_axis = np.arange(config['L'])
    m_axis = t_axis[1:-1]

    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # 左图: G(t) 两点关联函数
        axs[0].set_box_aspect(1 / 1.6875)
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].errorbar(t_axis, G_mean.numpy(), yerr=G_err.numpy(), fmt='-o',
                        color='blue', ecolor='purple', capsize=4, elinewidth=1.5, label='Local Metropolis')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('G(t)')
        axs[0].set_title(f'2-point Function (L={config["L"]})')
        axs[0].grid(True, which="both", ls="--", alpha=0.5)
        axs[0].legend()

        # 右图: Effective Mass 有效质量平台
        axs[1].set_box_aspect(1 / 1.6875)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].errorbar(m_axis, m_mean.numpy(), yerr=m_err.numpy(), fmt='-s',
                        color='red', ecolor='purple', capsize=4, elinewidth=1.5, label='Effective Mass')
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('Mass')
        axs[1].set_title(f'Effective Mass (L={config["L"]})')

        # Y轴自适应边距处理
        y_min, y_max = m_mean.min().item(), m_mean.max().item()
        pad = (y_max - y_min) * 0.5 if y_max != y_min else 0.1
        axs[1].set_ylim(y_min - pad, y_max + pad)
        axs[1].grid(True, ls="--", alpha=0.7)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"绘图时发生错误: {e}")


if __name__ == "__main__":
    # 依次执行各尺寸的模拟
    for current_config in CONFIGS:
        main(current_config)