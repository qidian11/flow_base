import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 严格按照 Albergo 原论文公式 (25) 和 (26) 的计算与排版
# ==========================================

tick_labels = [r'$6^2$', r'$8^2$', r'$10^2$', r'$12^2$', r'$14^2$']
L_list = [6, 8, 10, 12, 14]
algorithms = ["Local", "HMC", "NF"]
N_samples = 1000000

results = {alg: {"chi2_mean": [], "chi2_err": [], "E_mean": [], "E_err": []} for alg in algorithms}


def compute_chi2_E(configs):
    """
    严格按照论文：
    chi2 = sum_x G_c(x)
    E = (1/d) sum_mu G_c(mu_hat)

    其中 G_c(x) = 1/V sum_y [
        <phi(y) phi(y+x)> - <phi(y)> <phi(y+x)>
    ]
    """
    _, L_dim, _ = configs.shape
    V = L_dim * L_dim
    d = 2

    # ---------- chi2 ----------
    # M_n = sum_x phi_n(x)
    M = np.sum(configs, axis=(1, 2))

    # connected susceptibility:
    # chi2 = <M^2>/V - <M>^2/V
    # 为了保留 sample-wise bootstrap，可以先构造 centered estimator
    M_centered = M - np.mean(M)
    chi2 = (M_centered ** 2) / V

    # ---------- E ----------
    phi_right = np.roll(configs, shift=-1, axis=2)
    phi_up = np.roll(configs, shift=-1, axis=1)

    # raw nearest-neighbor correlators per configuration
    G1_raw = np.sum(configs * phi_right, axis=(1, 2)) / V
    G2_raw = np.sum(configs * phi_up, axis=(1, 2)) / V

    # ensemble mean field at each site
    phi_mean = np.mean(configs, axis=0)

    # connected subtraction terms
    phi_mean_right = np.roll(phi_mean, shift=-1, axis=1)
    phi_mean_up = np.roll(phi_mean, shift=-1, axis=0)

    G1_sub = np.sum(phi_mean * phi_mean_right) / V
    G2_sub = np.sum(phi_mean * phi_mean_up) / V

    G1_conn = G1_raw - G1_sub
    G2_conn = G2_raw - G2_sub

    E = (G1_conn + G2_conn) / d

    return chi2, E


def bootstrap_error(data, bin_size=100, num_bootstraps=500):
    """
    Errors indicate 68% confidence intervals estimated using bootstrap resampling with bins of size 100.
    """
    N = len(data)
    num_bins = N // bin_size
    binned_data = np.mean(data[:num_bins * bin_size].reshape(num_bins, bin_size), axis=1)

    boot_means = np.zeros(num_bootstraps)
    for i in range(num_bootstraps):
        sample = np.random.choice(binned_data, size=num_bins, replace=True)
        boot_means[i] = np.mean(sample)

    return np.mean(binned_data), np.std(boot_means, ddof=1)


print("🚀 开始处理 FIG. 5 的数据计算...")
for alg in algorithms:
    for L in L_list:
        filename = f"{alg}_configs_L{L}_N1280000_DTYPE_double.npz" if alg != "NF" else f"NF_configs_L{L}_N1000000_DTYPE_double.npz"
        if not os.path.exists(filename):
            print(f"  [跳过] 找不到 {filename}")
            results[alg]["chi2_mean"].append(np.nan)
            results[alg]["chi2_err"].append(np.nan)
            results[alg]["E_mean"].append(np.nan)
            results[alg]["E_err"].append(np.nan)
            continue

        data = np.load(filename)
        configs = data['configs'] if 'configs' in data else data['arr_0']
        if configs.ndim == 4: configs = configs.reshape(configs.shape[0], configs.shape[-2], configs.shape[-1])

        configs = configs[:N_samples]
        chi2_ts, E_ts = compute_chi2_E(configs)

        c2_m, c2_e = bootstrap_error(chi2_ts, bin_size=100)
        e_m, e_e = bootstrap_error(E_ts, bin_size=100)

        results[alg]["chi2_mean"].append(c2_m)
        results[alg]["chi2_err"].append(c2_e)
        results[alg]["E_mean"].append(e_m)
        results[alg]["E_err"].append(e_e)

# ==========================================
# 画图阶段：匹配 PRD 期刊双栏长宽比 (12 x 4.5)
# ==========================================
plt.style.use("default")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

label_map = {"HMC": "HMC", "Local": "local Metropolis", "NF": "ML"}
marker_map = {"HMC": "o", "Local": "D", "NF": "s"}
color_map = {"HMC": "#1f77b4", "Local": "#ff7f0e", "NF": "#2ca02c"}
offsets = {"Local": 0.0, "HMC": 0.0, "NF": 0.0}

plot_order = ["HMC", "Local", "NF"]

for alg in plot_order:
    x_pos = np.array(L_list) + offsets[alg]

    # 纯线性系，带趋势连线
    axes[0].errorbar(x_pos, results[alg]["chi2_mean"], yerr=results[alg]["chi2_err"],
                     fmt=marker_map[alg], color=color_map[alg], label=label_map[alg],
                     linestyle='-', linewidth=1.5, alpha=0.9,
                     capsize=3, markersize=7, elinewidth=1.5)

    axes[1].errorbar(x_pos, results[alg]["E_mean"], yerr=results[alg]["E_err"],
                     fmt=marker_map[alg], color=color_map[alg], label=label_map[alg],
                     linestyle='-', linewidth=1.5, alpha=0.9,
                     capsize=3, markersize=7, elinewidth=1.5)

# --- 坐标轴格式化 ---
axes[0].set_ylabel(r"Susceptibility $\chi_2$", fontsize=14)
axes[0].set_xlabel(r"$L$", fontsize=14)
axes[0].set_xticks(L_list)
axes[0].set_xticklabels(tick_labels)
axes[0].legend(fontsize=12, loc="upper left", frameon=False)

axes[1].set_ylabel(r"Ising energy $E$", fontsize=14)
axes[1].set_xlabel(r"$V$", fontsize=14)
axes[1].set_xticks(L_list)
axes[1].set_xticklabels(tick_labels)

# 学术规范：刻度朝内
for ax in axes:
    ax.tick_params(axis='both', which='major', direction='in', length=6, labelsize=12)
    ax.grid(False)

plt.tight_layout(pad=2.0)
plt.savefig("FIG5_Albergo_Perfect_Reproduction.png", bbox_inches='tight')
print("🎉 FIG. 5 已按照原论文公式与期刊版式完美生成！")
plt.show()