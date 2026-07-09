import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
from scipy.optimize import curve_fit

# ==========================================
# 1. 配置与高度可自定义的绘图参数 (PLOT_CONFIG)
# ==========================================
PLOT_CONFIG = {
    "style": "seaborn-v0_8-ticks",  # Matplotlib 样式主题
    "figure_size": (13, 6),  # 整体画布大小 (宽, 高)
    "dpi": 300,  # 图像分辨率

    # 磁化率与能量图的自定义配置
    "chi2": {
        "title": r"Critical Slowing Down: Susceptibility $\chi_2$",
        "xlabel": r"Lattice Size $L$",
        "ylabel": r"Integrated Autocorrelation Time $\tau_{int}(\chi_2)$",
        "x_limits": (5, 16),  # X轴范围 [xmin, xmax]
        "y_limits": None,  # Y轴范围 (设为 None 则自动适应)
    },
    "E": {
        "title": r"Critical Slowing Down: Ising Energy $E$",
        "xlabel": r"Lattice Size $L$",
        "ylabel": r"Integrated Autocorrelation Time $\tau_{int}(E)$",
        "x_limits": (5, 16),
        "y_limits": None,
    },

    # 各算法的线条、颜色、标记与标签 (可在内部随意调整)
    "algorithms": {
        "Local": {"color": "#E63946", "marker": "o", "linestyle": "-", "linewidth": 2.5, "label": "Local Metropolis"},
        "HMC": {"color": "#457B9D", "marker": "s", "linestyle": "--", "linewidth": 2.5, "label": "Hybrid Monte Carlo"},
        "NF": {"color": "#2A9D8F", "marker": "^", "linestyle": "-.", "linewidth": 2.5,
               "label": "Normalizing Flow (IMH)"}
    },

    # 拟合虚线的全局样式
    "fit_line": {"linestyle": ":", "linewidth": 1.5, "alpha": 0.7},

    # 全局字体大小控制
    "fontsize_title": 14,
    "fontsize_label": 12,
    "fontsize_tick": 11,
    "fontsize_legend": 10
}

# 硬件设备自动选择 (支持 CUDA 和 Intel XPU 架构)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
)
print(f"🔥 当前数据生成与计算设备: {DEVICE}")


# ==========================================
# 2. Normalizing Flow 模型动态加载与数据生成
# ==========================================
# ==========================================
# 2. Normalizing Flow 模型动态加载与数据生成 (完美适配版)
# ==========================================
def load_nf_model_and_action(L):
    """
    动态导入对应的 Python 脚本，完美复用你源码中的一切类、函数和自动寻重机制。
    """
    if L == 14:
        script_name = "final_normalizing.py"
    else:
        script_name = f"final_normalizing_L{L}.py"

    if not os.path.exists(script_name):
        raise FileNotFoundError(f"找不到模型定义脚本: {script_name}")

    # 动态加载你的模块
    spec = importlib.util.spec_from_file_location("nf_module", script_name)
    nf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf_module)

    CONFIG = nf_module.CONFIG

    # 1. 完美对接你的模型类 (FlowModel)
    model = nf_module.FlowModel(CONFIG).to(DEVICE).to(torch.float64)

    # 2. 完美对接你的先验分布 (兼容 L14 的硬编码 m_sq_prior 和 L6 的字典配置)
    m_sq_prior = CONFIG.get('m_sq_prior', 0.6005269985)
    prior = nf_module.FreeFieldPrior(CONFIG['L'], m_sq_prior).to(DEVICE).to(torch.float64)

    # 3. 完美复刻你的权重搜寻逻辑
    # 你的源码在达标后会保存为: 'mcmc_success_' + checkpoint_path
    target_success_weight = "mcmc_success_" + CONFIG['checkpoint_path']

    if os.path.exists(target_success_weight):
        final_weight_path = target_success_weight
        print(f"  --> 🎯 找到成功达标权重: {final_weight_path}")
    else:
        # 如果没有成功标志的，直接调用你源码里的自动搜寻函数找最新的
        final_weight_path = nf_module.auto_find_latest_checkpoint(CONFIG)
        if final_weight_path:
            print(f"  --> 🔍 未找到 success 权重，调用内置搜索找到最新断点: {final_weight_path}")

    # 4. 加载权重
    if final_weight_path and os.path.exists(final_weight_path):
        checkpoint = torch.load(final_weight_path, map_location=DEVICE, weights_only=False)
        # 兼容你源码中可能的 compile 前缀
        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(clean_dict)
    else:
        print(f"  [严重警告] 彻底没找到任何 L={L} 的权重，将使用随机初始化的模型！")

    # 5. 提取你源码中的作用量计算函数
    action_fn = nf_module.compute_action

    return model, prior, action_fn, CONFIG


def generate_nf_mcmc_data(L, n_samples=1000000, batch_size=10000):
    """
    生成数据，逻辑参考了你源码中的 CPU/GPU 混合提速 MCMC。
    """
    model, prior, action_fn, CONFIG = load_nf_model_and_action(L)
    model.eval()

    configurations = np.zeros((n_samples, L, L), dtype=np.float64)

    dummy_progress = torch.tensor(1.0, device=DEVICE, dtype=torch.float64)
    enforce_sym = CONFIG.get('enforce_z2_sym', False)

    print(f"🚀 开始为 L={L} 生成 Normalizing Flow MCMC 链...")

    with torch.no_grad():
        # --- 初始状态 ---
        z_curr, log_p_z_curr = prior.sample(1)
        phi_curr, log_det_J_curr = model(z_curr, dummy_progress, enforce_sym=enforce_sym)
        log_q_curr = log_p_z_curr - log_det_J_curr
        S_curr = action_fn(phi_curr)

        accepted_count = 0

        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)

            # 批量提议
            z_prop, log_p_z_prop = prior.sample(current_batch)
            phi_prop, log_det_J_prop = model(z_prop, dummy_progress, enforce_sym=enforce_sym)

            log_q_prop = log_p_z_prop - log_det_J_prop
            S_prop = action_fn(phi_prop)

            # --- 借用你源码里的 CPU 标量判别法提速 ---
            S_np = S_prop.cpu().numpy()
            log_q_np = log_q_prop.cpu().numpy()

            curr_S_val = S_curr.item()
            curr_log_q_val = log_q_curr.item()

            log_rands = np.log(np.random.rand(current_batch))

            for j in range(current_batch):
                prop_S_val = S_np[j]
                prop_log_q_val = log_q_np[j]

                log_acc_ratio = (-prop_S_val - prop_log_q_val) - (-curr_S_val - curr_log_q_val)

                if log_rands[j] < log_acc_ratio:
                    # 接受，更新标量与张量状态
                    curr_S_val = prop_S_val
                    curr_log_q_val = prop_log_q_val
                    phi_curr = phi_prop[j:j + 1]
                    S_curr = S_prop[j:j + 1]
                    log_q_curr = log_q_prop[j:j + 1]
                    accepted_count += 1

                # 无论接受与否，写入构型
                configurations[i + j] = phi_curr.cpu().numpy().reshape(L, L)

            if (i + current_batch) % 200000 == 0 or (i + current_batch) == n_samples:
                print(
                    f"  进度: {i + current_batch}/{n_samples} | 累计接受率: {accepted_count / (i + current_batch):.2%}")

    output_filename = f"NF_configs_L{L}_N1000000_DTYPE_double.npz"
    np.savez(f"./{output_filename}", configs=configurations)
    print(f"✅ L={L} 的 NF 数据生成完毕，已固化保存至: {output_filename}\n")
    return configurations


# ==========================================
# 3. 物理可观测量与自相关时间 (Wolff 窗口法)
# ==========================================
def calculate_observables(configs):
    """
    高效计算磁化率 proxy (chi_2) 和 Ising-like 能量 (E) 序列
    """
    # 🌟 新增：防御性降维，兼容 (N, 1, L, L) 或 (N, L, L, 1) 的情况
    if configs.ndim == 4:
        configs = configs.reshape(configs.shape[0], configs.shape[-2], configs.shape[-1])

    _, L, _ = configs.shape
    V = L * L

    # 磁化率 chi_2 = M^2 / V
    M = np.sum(configs, axis=(1, 2))
    chi2 = (M ** 2) / V

    # 能量 E (最近邻项)
    phi_x_right = np.roll(configs, shift=-1, axis=2)
    phi_x_up = np.roll(configs, shift=-1, axis=1)
    E = np.sum(configs * phi_x_right + configs * phi_x_up, axis=(1, 2)) / V
    return chi2, E


def integrated_autocorr_time(x, c=5.0):
    """
    使用 Ulli Wolff 自动窗口法截断计算积分自相关时间
    """
    N = len(x)
    x = x - np.mean(x)
    C0 = np.var(x, ddof=1)
    if C0 == 0: return 0.5

    tau_int = 0.5
    for t in range(1, N):
        Ct = np.mean(x[:N - t] * x[t:])
        rho_t = Ct / C0
        tau_int += rho_t
        if t >= c * tau_int:  # Madras-Sokal 截断基准
            break
    return max(0.5, tau_int)


# ==========================================
# 4. 主循环：读取、切片、计算与 Log-Log 绘图
# ==========================================
def run_scaling_analysis():
    L_list = [6, 8, 10, 12, 14]
    algorithms = ["Local", "HMC", "NF"]

    # 结果容器
    results = {alg: {"chi2": [], "E": []} for alg in algorithms}

    for alg in algorithms:
        for L in L_list:
            if alg == "NF":
                # 检查本地是否有现成的 NF 1M 数据，没有则在线实时触发生成
                filename = f"NF_configs_L{L}_N1000000_DTYPE_double.npz"
                if not os.path.exists(filename):
                    try:
                        configs = generate_nf_mcmc_data(L, n_samples=1000000)
                    except Exception as e:
                        print(f"  [跳过] 无法在线生成 L={L} 的 NF 数据，原因: {e}")
                        results[alg]["chi2"].append(np.nan)
                        results[alg]["E"].append(np.nan)
                        continue
                else:
                    print(f"  加载现有数据: {filename}")
                    data = np.load(filename)
                    configs = data['configs']
            else:
                # Local 与 HMC 原始文件包含 1,280,000 个数据
                filename = f"{alg}_configs_L{L}_N1280000_DTYPE_double.npz"
                if os.path.exists(filename):
                    print(f"  加载数据并严格切片 [0:1000000]: {filename}")
                    data = np.load(filename)
                    # 兼容不同写入 key 值的读取
                    raw_configs = data['configs'] if 'configs' in data else data['arr_0']
                    # 🌟 关键点：丢弃后 280,000 个，只截取前 1,000,000 个
                    configs = raw_configs[:1000000]
                else:
                    print(f"  [警告] 未能找到文件: {filename}")
                    results[alg]["chi2"].append(np.nan)
                    results[alg]["E"].append(np.nan)
                    continue

            # 计算对应的热力学可观测量序列
            chi2_series, E_series = calculate_observables(configs)

            # 统计计算自相关时间
            tau_chi2 = integrated_autocorr_time(chi2_series)
            tau_E = integrated_autocorr_time(E_series)

            results[alg]["chi2"].append(tau_chi2)
            results[alg]["E"].append(tau_E)
            print(f"    --> {alg} (L={L}): \\tau_int(chi2) = {tau_chi2:.2f}, \\tau_int(E) = {tau_E:.2f}")

    # --- 动态学术图表绘制阶段 ---
    plt.style.use(PLOT_CONFIG["style"])
    fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"])

    def plot_sub_panel(ax, obs_key, cfg_axes):
        ax.set_title(cfg_axes["title"], fontsize=PLOT_CONFIG["fontsize_title"])
        ax.set_xlabel(cfg_axes["xlabel"], fontsize=PLOT_CONFIG["fontsize_label"])
        ax.set_ylabel(cfg_axes["ylabel"], fontsize=PLOT_CONFIG["fontsize_label"])

        # 强制切换为标准双对数（Log-log）坐标轴
        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)

        # 应用用户自定义的坐标轴区间限制
        if cfg_axes["x_limits"]: ax.set_xlim(cfg_axes["x_limits"])
        if cfg_axes["y_limits"]: ax.set_ylim(cfg_axes["y_limits"])

        # 优化刻度展现形式：使 X 轴上的标度直接显示离散的晶格尺寸数值而非科学计数法
        ax.set_xticks(L_list)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(labelsize=PLOT_CONFIG["fontsize_tick"])

        for alg in algorithms:
            y_data = np.array(results[alg][obs_key])
            valid_mask = ~np.isnan(y_data)
            if np.sum(valid_mask) == 0: continue

            x_plot = np.array(L_list)[valid_mask]
            y_plot = y_data[valid_mask]
            style_cfg = PLOT_CONFIG["algorithms"][alg]

            # 绘制真实观测离散点与实线
            main_line, = ax.plot(x_plot, y_plot,
                                 color=style_cfg["color"],
                                 marker=style_cfg["marker"],
                                 linestyle=style_cfg["linestyle"],
                                 linewidth=style_cfg["linewidth"],
                                 markersize=7,
                                 label=style_cfg["label"])

            # 自动拟合临界慢化指数 z (\tau = a * L^z)
            if len(x_plot) >= 3:
                popt = np.polyfit(np.log10(x_plot), np.log10(y_plot), 1)
                z_exponent = popt[0]

                # 绘制延长拟合虚线
                fit_x = np.linspace(min(x_plot) * 0.95, max(x_plot) * 1.05, 100)
                fit_y = (10 ** popt[1]) * (fit_x ** z_exponent)
                ax.plot(fit_x, fit_y,
                        color=style_cfg["color"],
                        linestyle=PLOT_CONFIG["fit_line"]["linestyle"],
                        linewidth=PLOT_CONFIG["fit_line"]["linewidth"],
                        alpha=PLOT_CONFIG["fit_line"]["alpha"])

                # 更新图例，直接将动态拟合得到的临界指数 z 渲染至 label 内
                main_line.set_label(f"{style_cfg['label']} ($z \\approx {z_exponent:.2f}$)")

        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(fontsize=PLOT_CONFIG["fontsize_legend"], loc="upper left", frameon=True, edgecolor='gray')

    # 渲染左图 (\chi2) 和 右图 (E)
    plot_sub_panel(axes[0], "chi2", PLOT_CONFIG["chi2"])
    plot_sub_panel(axes[1], "E", PLOT_CONFIG["E"])

    plt.tight_layout()
    output_img = "CSD_Scaling_1M_Comparison.png"
    plt.savefig(output_img, bbox_inches='tight')
    print(f"\n🎉 标度图表绘制成功！已导出高分辨率图像至: {output_img}")
    plt.show()


if __name__ == "__main__":
    run_scaling_analysis()