import numpy as np
import matplotlib.pyplot as plt


def compute_observable_series(configs):
    """
    计算特定物理量的时间序列。
    这里以全局平均场平方（代表磁化率 chi_2 的简化代理）为例。
    输入 configs 维度: (N, 1, L, L)
    输出 shape: (N,)
    """
    print("正在计算物理量序列...")
    # 对每个构型的空间维度求平均
    mean_field = np.mean(configs, axis=(1, 2, 3))
    # 取平方作为观测物理量
    observable = mean_field ** 2
    return observable


def compute_autocorr_function(O_series, tau_max):
    """
    计算标准物理量的自相关函数 rho(tau) / rho(0)
    对应论文公式 (17)
    """
    print(f"正在计算标准自相关函数 (最大步长 tau_max={tau_max})...")
    N = len(O_series)
    mean_O = np.mean(O_series)
    var_O = np.var(O_series)  # 也就是 rho(0)

    rho_norm = np.zeros(tau_max)
    for tau in range(tau_max):
        # 计算协方差
        cov = np.mean((O_series[:N - tau] - mean_O) * (O_series[tau:] - mean_O))
        rho_norm[tau] = cov / var_O

    return rho_norm


def compute_integrated_autocorr(rho_norm, c=5.0):
    """
    计算积分自相关时间 tau_int
    加入自动截断窗口 (Madras-Sokal windowing) 防止大 tau 时噪声指数级放大
    """
    tau_int = 0.5
    for tau in range(1, len(rho_norm)):
        tau_int += rho_norm[tau]
        # 截断条件：当累加的窗口大小超过 tau_int 的 c 倍时停止
        if tau >= c * tau_int:
            break
    return tau_int


def compute_rejection_estimator(accept_history, tau_max):
    """
    计算基于拒绝率的自相关估计 (Observable-independent Estimator)
    对应论文公式 (15) 和 (16)
    """
    print("正在计算基于拒绝连续性的自相关估计...")
    N = len(accept_history)
    # 转换为拒绝历史 (True 表示被拒绝)
    reject_history = ~accept_history

    y_tau_rej = np.zeros(tau_max)
    y_tau_rej[0] = 1.0  # tau=0 时必然是 1

    for tau in range(1, tau_max):
        # 寻找连续 tau 次被拒绝的序列
        # 使用 numpy 的滑动窗口按位与 (bitwise AND)
        consecutive_rejects = np.ones(N - tau, dtype=bool)
        for t in range(tau):
            consecutive_rejects &= reject_history[t:N - tau + t]

        y_tau_rej[tau] = np.mean(consecutive_rejects)

    return y_tau_rej


if __name__ == "__main__":
    # ==========================================
    # 1. 加载你的生成数据
    # ==========================================
    # 请替换为你的实际文件路径
    file_path = "phi_ensemble_prior_cnn_res_model_double_precision_False_14_loss_history_coupling_layers_12_multi_k_3_dil_1_depth_2_layers_12cnn_0attn_hidden_layers_4_hidden_channels_256_iterations_30000.npz"
    try:
        data = np.load(file_path)
        configs = data['configs']
        accept_history = data['accept_history']
        print(f"数据加载成功！总构型数: {len(configs)}")
    except FileNotFoundError:
        print(f"未找到文件 {file_path}，请确保路径正确。")
        exit()

    # 设置我们要观察的最大迟滞步数 tau
    TAU_MAX = 100

    # ==========================================
    # 2. 计算物理量自相关时间 (Standard Estimator)
    # ==========================================
    O_series = compute_observable_series(configs)
    rho_norm = compute_autocorr_function(O_series, TAU_MAX)
    tau_int_O = compute_integrated_autocorr(rho_norm)
    print(f"✅ 物理量的积分自相关时间 (tau_int) 约为: {tau_int_O:.4f}")

    # ==========================================
    # 3. 计算无物理量依赖的自相关估计 (Rejection Estimator)
    # ==========================================
    y_tau_rej = compute_rejection_estimator(accept_history, TAU_MAX)
    tau_int_rej = compute_integrated_autocorr(y_tau_rej)
    print(f"✅ 基于 MH 拒绝率的积分自相关时间预估为: {tau_int_rej:.4f}")

    # ==========================================
    # 4. 可视化对比 (可选，推荐)
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(range(TAU_MAX), rho_norm, label=r'Observable $\rho_{\mathcal{O}}(\tau)/\rho_{\mathcal{O}}(0)$', marker='o',
             markersize=4)
    plt.plot(range(TAU_MAX), y_tau_rej, label=r'Rejection Estimator $y_{\tau rej}$', linestyle='--', color='red',
             linewidth=2)

    plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel(r'Markov Chain Separation $\tau$')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Decay Analysis')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('autocorr_analysis.png')
    print("图表已保存为 autocorr_analysis.png")