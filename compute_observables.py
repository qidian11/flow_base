import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from HMC_final import CONFIG
from prior_cnn_macro_z2 import CONFIG
# from aligned_prior_cnn import CONFIG

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


device = get_device(prefer="auto")
print(f"🔥 当前计算设备: {device}")

DTYPE = torch.float64


# ==========================================
# 纯双循环 (Double Loop) + Super Binning + Bootstrap
# ==========================================

def compute_observables_with_error(ensemble_tensor, bin_size, boot_time):
    """
    输入 shape: [N_samples, batchsize, L_space, L_time]
    使用最原汁原味的物理空间平移双循环，替代 FFT 逻辑。
    """
    N_samples, batchsize, L_s, L_t = ensemble_tensor.shape
    device = ensemble_tensor.device

    print("\n--- 开始执行高能物理统计管线 (纯双循环版) ---")

    n_bins = N_samples // bin_size
    total_bins = n_bins * batchsize

    # 截断无法凑齐一整个 bin 的零碎样本
    ensemble_trunc = ensemble_tensor[:n_bins * bin_size]  # [N_trunc, batchsize, L_s, L_t]

    # ---------------------------------------------------------
    # Step 1: 提取单构型磁化强度 M 并 Binning
    # ---------------------------------------------------------
    print("1. 正在计算单构型磁化强度 M 并进行 Binning...")
    M_all = ensemble_trunc.mean(dim=(2, 3))  # [N_trunc, batchsize]
    # 先在时间序列上求 bin 均值，然后打平多链维度 (Super Binning)
    M_binned = M_all.view(n_bins, bin_size, batchsize).mean(dim=1)
    M_binned = M_binned.view(total_bins)

    # ---------------------------------------------------------
    # 【新增】Step 1.5: 提取 phi^1 到 phi^5 并进行 Binning
    # ---------------------------------------------------------
    print("1.5 正在计算 phi^1 到 phi^5 的期望值并进行 Binning...")
    phi_pow_binned_list = []
    for power in range(1, 6):
        if power == 1:
            # phi^1 就是上面的 M_binned，直接复用避免重复计算
            phi_pow_binned_list.append(M_binned)
        else:
            pow_all = (ensemble_trunc ** power).mean(dim=(2, 3))
            pow_binned = pow_all.view(n_bins, bin_size, batchsize).mean(dim=1).view(total_bins)
            phi_pow_binned_list.append(pow_binned)

    # 打包成一个张量，shape: [total_bins, 5]
    phi_pow_binned_tensor = torch.stack(phi_pow_binned_list, dim=1)

    # ---------------------------------------------------------
    # Step 2: 双循环计算关联函数 + 零动量投影 + Binning
    # ---------------------------------------------------------
    print("2. 正在通过双循环平移计算两点关联函数 (这比FFT略慢，请耐心等待)...")
    G_unc_binned_list = []

    # 针对每个时间间隔 t
    for t in range(L_t):
        corr_l_list = []
        # 针对每个空间间隔 l
        for l in range(L_s):
            # 沿空间(dim=2)和时间(dim=3)进行 PBC 平移
            shifted_ensemble = torch.roll(ensemble_trunc, shifts=(l, t), dims=(2, 3))

            # 单构型内部点乘，并求全空间体积均值
            # (物理含义：计算 1/V \sum_x \phi(x)\phi(x+shift) )
            corr = (shifted_ensemble * ensemble_trunc).mean(dim=(2, 3))  # [N_trunc, batchsize]

            # 对 MCMC 时间序列进行 Binning 去相关
            corr_binned = corr.view(n_bins, bin_size, batchsize).mean(dim=1)  # [n_bins, batchsize]

            corr_l_list.append(corr_binned)

        # 将当前时间间隔 t 下，所有空间偏移 l 的结果打包
        corr_all_l = torch.stack(corr_l_list, dim=-1)  # [n_bins, batchsize, L_s]

        # 零动量投影：沿空间维度(dim=-1)严格求和，得到 G(t)
        G_unc_t_binned = corr_all_l.mean(dim=-1)  # [n_bins, batchsize]

        # 展平合并所有平行链的 Bins (Super Binning)
        G_unc_t_binned = G_unc_t_binned.view(total_bins)

        G_unc_binned_list.append(G_unc_t_binned)

    # 打包所有时间间隔 t，得到最终的 binned 基础非连通矩阵
    G_unc_binned = torch.stack(G_unc_binned_list, dim=-1)  # [total_bins, L_t]

    print(f"   => 成功生成了 {total_bins} 个绝对独立的统计砖块！")

    # ---------------------------------------------------------
    # Step 3: 全样本提取官方中心值 (对应论文图 3, 4 的数据点)
    # ---------------------------------------------------------
    G_unc_full_mean = G_unc_binned.mean(dim=0)
    M_full_mean = M_binned.mean()
    # 【新增】phi^1 到 phi^5 的中心值, shape: [5]
    phi_pow_central = phi_pow_binned_tensor.mean(dim=0)

    # 连通图 = 非连通图 - 零动量背景 = 非连通图 - L_s * <phi>^2
    G_conn_central = G_unc_full_mean - L_s * (M_full_mean ** 2)

    cosh_m_central = (torch.roll(G_conn_central, shifts=1, dims=0) +
                      torch.roll(G_conn_central, shifts=-1, dims=0)) / (2 * G_conn_central)
    cosh_m_central = torch.clamp(cosh_m_central, min=1.000001)
    m_eff_central = torch.acosh(cosh_m_central)

    # ---------------------------------------------------------
    # Step 4: Bootstrap 测量真实涨落 (延迟一切非线性计算！)
    # ---------------------------------------------------------
    print(f"3. Bootstrap 重组宇宙 {boot_time} 次以提取真实误差棒...")
    G_conn_boot_list = []
    m_eff_boot_list = []
    phi_pow_boot_list = []  # 【新增】用于收集每次 bootstrap 的 phi^n

    for _ in range(boot_time):
        # 每次有放回地抽取相互独立的宇宙块
        idx = torch.randint(0, total_bins, (total_bins,), device=device)

        # 1. 抽取当前宇宙的基础量
        G_unc_star = G_unc_binned[idx].mean(dim=0)
        M_star = M_binned[idx].mean()

        # 【新增】抽取当前的 phi^n 期望
        phi_pow_star = phi_pow_binned_tensor[idx].mean(dim=0)
        phi_pow_boot_list.append(phi_pow_star)

        # 2. 减去真空断开图，得到连通信号
        G_conn_star = G_unc_star - L_s * (M_star ** 2)

        # 3. 计算有效质量 (必须置于 Bootstrap 内部)
        G_t_minus_1 = torch.roll(G_conn_star, shifts=1, dims=0)
        G_t_plus_1 = torch.roll(G_conn_star, shifts=-1, dims=0)
        cosh_m_star = (G_t_minus_1 + G_t_plus_1) / (2 * G_conn_star)
        cosh_m_star = torch.clamp(cosh_m_star, min=1.000001)  # 防崩溃底线
        m_eff_star = torch.acosh(cosh_m_star)

        G_conn_boot_list.append(G_conn_star)
        m_eff_boot_list.append(m_eff_star)

    # 计算标准差作为误差棒
    G_conn_err = torch.stack(G_conn_boot_list).std(dim=0)
    m_eff_err = torch.stack(m_eff_boot_list).std(dim=0)
    phi_pow_err = torch.stack(phi_pow_boot_list).std(dim=0)  # 【新增】phi^n 的误差棒

    print("--- 物理统计管线执行完毕 ---\n")
    return G_conn_central, G_conn_err, m_eff_central, m_eff_err, phi_pow_central, phi_pow_err


# ==========================================
# 模块 3: 主流程与可视化
# ==========================================

def main(ensemble_tensor, bin_size=100, boot_time=2000):
    print(f"Raw Ensemble Shape: {ensemble_tensor.shape}")

    # 分析矩阵可以放回 GPU 加速计算
    ensemble_tensor = ensemble_tensor.to(device)

    # 【更新】接收额外返回的 phi_pow 数据
    G_t_mean, G_t_err, m_eff_mean, m_eff_err, phi_pow_mean, phi_pow_err = compute_observables_with_error(
        ensemble_tensor,
        bin_size=bin_size,
        boot_time=boot_time
    )

    # 转移回 CPU 用于 matplotlib 绘图
    G_t_mean = G_t_mean.cpu().numpy()
    G_t_err = G_t_err.cpu().numpy()
    m_eff_mean = m_eff_mean.cpu().numpy()
    m_eff_err = m_eff_err.cpu().numpy()
    phi_pow_mean = phi_pow_mean.cpu().numpy()
    phi_pow_err = phi_pow_err.cpu().numpy()

    # 将计算结果打包保存为 npz 文件
    # 文件名自动带上当前的晶格尺寸 L
    save_filename = f"{CONFIG['type']}_observables_result_L{CONFIG['L']}_double_precision_{CONFIG['double_precision']}.npz"
    np.savez_compressed(
        save_filename,
        G_t_mean=G_t_mean,
        G_t_err=G_t_err,
        m_eff_mean=m_eff_mean,
        m_eff_err=m_eff_err,
        phi_pow_mean=phi_pow_mean,
        phi_pow_err=phi_pow_err,
        L=CONFIG['L']  # 顺便把 L 也存进去，画图的时候方便用
    )

    print("\n" + "=" * 40)
    print(f"🎉 计算完成！所有统计数据已成功打包保存至: {save_filename}")
    print("=" * 40 + "\n")

    print("\n========== 数值结果 ==========\n")
    # 【新增】打印 phi^1 到 phi^5 的期望值与误差 (保留 6 位小数)
    print("------ Phi Powers Expectation ------")
    for power in range(1, 6):
        print(f"phi^{power}: {phi_pow_mean[power - 1]:.6f} ± {phi_pow_err[power - 1]:.6f}")
    print("-" * 36 + "\n")

    print("t    G_c(t)              error")
    for t in range(len(G_t_mean)):
        print(f"{t:2d}   {G_t_mean[t]:.6e}   ± {G_t_err[t]:.6e}")

    print("\n------ Effective Mass ------\n")
    print("t    m_eff(t)           error")
    for t in range(len(m_eff_mean)):
        print(f"{t:2d}   {m_eff_mean[t]:.6e}   ± {m_eff_err[t]:.6e}")

    # ================= 绘图部分 =================
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图：连通格林函数 G_c(t) ---
    axs[0].set_box_aspect(1 / 1.4)  # 论文常见比例
    t_axis = np.arange(CONFIG['L'])

    axs[0].errorbar(
        t_axis, G_t_mean, yerr=G_t_err,
        fmt='-o', color='darkblue', ecolor='purple', capsize=4, elinewidth=1.5,
        label='pre-sampling Lattice Data', markersize=4
    )
    axs[0].set_yscale('log')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_xlabel('Time Separation (t)')
    axs[0].set_ylabel(r'$\tilde{G}_c(0, t)$')
    axs[0].set_title('Connected Two-point Green\'s Function')
    axs[0].grid(True, which="both", ls="--", alpha=0.3)
    axs[0].legend()

    # --- 右图：有效质量 m_eff(t) ---
    axs[1].set_box_aspect(1 / 1.4)
    # 因为使用 roll 计算质量，t=0 和 t=L-1 跨越了 PBC 边界，物理上存在镜像干扰
    # 取中间部分
    slice_idx = slice(1, -1)
    t_axis_m = t_axis[slice_idx]
    m_mean_slice = m_eff_mean[slice_idx]
    m_err_slice = m_eff_err[slice_idx]

    axs[1].errorbar(
        t_axis_m, m_mean_slice, yerr=m_err_slice,
        fmt='-o', color='darkred', ecolor='purple', capsize=4, elinewidth=1.5,
        label=r'$m_p^{\rm eff}$', markersize=4
    )
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_xlabel('Time Separation (t)')
    axs[1].set_ylabel(r'$m_p^{\rm eff}(t)$')
    axs[1].set_title('Effective Pole Mass')
    axs[1].grid(True, which="major", ls="--", alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # CONFIG= CONFIG[-1]
    ensemble_path = CONFIG['phi_ensemble_save_path']
    data = np.load(ensemble_path)
    # 假设你之前存的是 npz 文件中的 'configs'
    loaded_configs = data['configs']
    ensemble_tensor = torch.from_numpy(loaded_configs).to(dtype=DTYPE)
    main(ensemble_tensor)