import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. 动态导入 CONFIG (直接读取你的代码配置)
# ==========================================
# 根据你需要分析的数据，取消注释对应的导入项：
# from HMC_final import CONFIG
from final_normalizing import CONFIG

# from var_shared_trunk_prior import CONFIG

# 统一格式：兼容 final_normalizing 的单个 dict 和 HMC_final 的 dict 列表
config_list = CONFIG if isinstance(CONFIG, list) else [CONFIG]


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
    return torch.device("cpu")


device = get_device()
print(f"🔥 当前计算设备: {device}")


def process_and_compute(config):
    """根据读取到的 config 提取参数并进行物理量计算"""
    L = config.get('L')
    # 优先读取字典里的 phi_ensemble_save_path
    ensemble_path = config.get('phi_ensemble_save_path')
    bin_size = config.get('bin_size', 100)
    boot_time = config.get('bootstrap_time', 2000)

    if not ensemble_path or not os.path.exists(ensemble_path):
        print(f"⚠️ 找不到构型文件 {ensemble_path} 或配置中未设置，已跳过 L={L}。")
        return

    # 自定义保存路径，沿用 config 里的 base_name (如果有的话)
    base_name = config.get('base_name', f"L{L}")
    save_path = f"observables_result_{base_name}.npz"

    print(f"\n{'=' * 50}")
    print(f"🚀 开始分析: L={L} | 数据源: {ensemble_path}")

    # ==========================================
    # 2. 读取构型数据
    # ==========================================
    try:
        data = np.load(ensemble_path)
        configs_np = data['configs']
        # 统一转为 [N, L_t, L_s] 形式
        configs = torch.tensor(configs_np, dtype=torch.float64, device=device)
        if configs.ndim == 4:
            configs = configs.squeeze(1)
    except Exception as e:
        print(f"⚠️ 读取失败: {e}")
        return

    N = configs.shape[0]
    n_bins = N // bin_size
    if n_bins < 2:
        print("⚠️ 样本量不足以进行 Binning 和 Bootstrap。")
        return

    configs = configs[:n_bins * bin_size]
    print(f"📊 有效样本数: {n_bins * bin_size} | Bins 数量: {n_bins}")

    # ==========================================
    # 3. 极速物理量计算 (FFT 降维打击)
    # ==========================================
    # 利用卷积定理: C(x,t) = IFFT(|FFT(phi)|^2) / V
    configs_k = torch.fft.fft2(configs)
    power_spec = torch.abs(configs_k) ** 2
    C_i = torch.fft.ifft2(power_spec).real / (L * L)

    M_i = configs.mean(dim=(1, 2))

    # 先 Binning，消除时间序列自相关
    C_bin = C_i.view(n_bins, bin_size, L, L).mean(dim=1)
    M_bin = M_i.view(n_bins, bin_size).mean(dim=1)

    # ==========================================
    # 4. Bootstrap 误差评估
    # ==========================================
    boot_G_tilde = torch.zeros((boot_time, L), dtype=torch.float64, device=device)
    boot_m_eff = torch.zeros((boot_time, L), dtype=torch.float64, device=device)
    boot_chi2 = torch.zeros(boot_time, dtype=torch.float64, device=device)
    boot_E = torch.zeros(boot_time, dtype=torch.float64, device=device)

    print(f"🔄 正在重采样宇宙 {boot_time} 次以提取真实误差棒...")
    for i in range(boot_time):
        idx = torch.randint(0, n_bins, (n_bins,), device=device)

        C_boot = C_bin[idx].mean(dim=0)  # [L, L]
        M_boot = M_bin[idx].mean(dim=0)  # scalar

        # 核心公式：连通格林函数 G_c(x, t)
        G_c = C_boot - M_boot ** 2

        # (1) 零动量格林函数 \tilde{G}_c(0, t)：对空间维度（假设 dim=1）求和
        G_tilde = G_c.sum(dim=1)
        boot_G_tilde[i] = G_tilde

        # (2) 两点磁化率 \chi_2：全时空积分
        boot_chi2[i] = G_c.sum()

        # (3) Ising 能量密度 E：相邻点相关性 1/2 * (G_c(1, 0) + G_c(0, 1))
        boot_E[i] = 0.5 * (G_c[0, 1] + G_c[1, 0])

        # (4) 有效极点质量 m_p^{eff}
        G_tilde_plus = torch.roll(G_tilde, shifts=-1, dims=0)
        G_tilde_minus = torch.roll(G_tilde, shifts=1, dims=0)
        ratio = (G_tilde_plus + G_tilde_minus) / (2 * G_tilde + 1e-12)
        ratio = torch.clamp(ratio, min=1.0000001)  # 绝对防御：防止 arccosh 崩溃
        boot_m_eff[i] = torch.acosh(ratio)

    # ==========================================
    # 5. 结果落盘与输出
    # ==========================================
    G_tilde_mean = boot_G_tilde.mean(dim=0).cpu().numpy()
    G_tilde_err = boot_G_tilde.std(dim=0).cpu().numpy()
    m_eff_mean = boot_m_eff.mean(dim=0).cpu().numpy()
    m_eff_err = boot_m_eff.std(dim=0).cpu().numpy()
    chi2_mean = boot_chi2.mean().item()
    chi2_err = boot_chi2.std().item()
    E_mean = boot_E.mean().item()
    E_err = boot_E.std().item()

    np.savez_compressed(
        save_path,
        L=L,
        G_tilde_mean=G_tilde_mean, G_tilde_err=G_tilde_err,
        m_eff_mean=m_eff_mean, m_eff_err=m_eff_err,
        chi2_mean=chi2_mean, chi2_err=chi2_err,
        E_mean=E_mean, E_err=E_err
    )

    print(f"\n🎉 物理观测量计算完毕！结果已保存至: {save_path}")
    print(f"   [磁化率 chi_2] = {chi2_mean:.6f} ± {chi2_err:.6f}")
    print(f"   [Ising 能量 E] = {E_mean:.6f} ± {E_err:.6f}")

    return G_tilde_mean, G_tilde_err, m_eff_mean, m_eff_err, L


# ==========================================
# 6. 主循环与可视化
# ==========================================
if __name__ == '__main__':
    for config in config_list:
        # 只处理那些显式指明包含生成构型的字典
        if 'phi_ensemble_save_path' in config and config['phi_ensemble_save_path']:
            result = process_and_compute(config)

            # --- 绘图展示 (对应每个处理完的 config) ---
            if result is not None:
                G_t_mean, G_t_err, m_eff_mean, m_eff_err, L = result

                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                t_axis = np.arange(L)

                # 左图：零动量连通格林函数
                axs[0].set_box_aspect(1 / 1.4)
                axs[0].errorbar(t_axis, G_t_mean, yerr=G_t_err, fmt='-o', color='darkblue', ecolor='purple', capsize=4,
                                label='Lattice Data')
                axs[0].set_yscale('log')
                axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
                axs[0].set_xlabel('Time Separation (t)')
                axs[0].set_ylabel(r'$\tilde{G}_c(0, t)$')
                axs[0].set_title(f'Connected Green\'s Function (L={L})')
                axs[0].grid(True, which="both", ls="--", alpha=0.3)
                axs[0].legend()

                # 右图：有效质量 (掐头去尾避免边界效应)
                axs[1].set_box_aspect(1 / 1.4)
                slice_idx = slice(1, -1)
                axs[1].errorbar(t_axis[slice_idx], m_eff_mean[slice_idx], yerr=m_eff_err[slice_idx], fmt='-o',
                                color='darkred', ecolor='purple', capsize=4, label=r'$m_p^{\rm eff}$')
                axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
                axs[1].set_xlabel('Time Separation (t)')
                axs[1].set_ylabel(r'$m_p^{\rm eff}(t)$')
                axs[1].set_title(f'Effective Pole Mass (L={L})')
                axs[1].grid(True, which="major", ls="--", alpha=0.3)
                axs[1].legend()

                plt.tight_layout()
                plt.show()