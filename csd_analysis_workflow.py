import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator, NullFormatter
import os
import re
import shutil
import importlib.util

# ==========================================
# 1. 绘图参数配置 (严苛纯净版 + 独立单图配置)
# ==========================================
PLOT_CONFIG = {
    "style": "seaborn-v0_8-ticks",
    "figure_size": (6, 5.5),  # 单张图的画布比例
    "dpi": 300,
    "fontsize_label": 18,
    "fontsize_tick": 15,
    "fontsize_legend": 13,

    # 强制统一的坐标轴范围与刻度
    "x_limits": (5.5, 14.5),
    "x_ticks": [6, 8, 10, 12, 14],
    "y_limits": (0.4, 8.0),
    "y_ticks": [0.5, 1, 2, 5],

    "markers": {
        "E": "o",
        "chi2": "s",
        "Gc0": "d"
    },
    "labels": {
        "E": r"$E$",
        "chi2": r"$\chi_2$",
        "Gc0": r"$G_c(0)$"
    },

    "colors": {
        "HMC": "black",
        "Local": "magenta",
        "NF": "#7CB342"
    }
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
)
print(f"🔥 当前计算设备: {DEVICE}")


# ==========================================
# 2. 核心 MCMC 引擎 (用于快速评估 & 百万生成)
# ==========================================
def run_mcmc_chain(model, prior, action_fn, n_samples, batch_size, enforce_sym, L):
    configurations = np.zeros((n_samples, L, L), dtype=np.float64)
    dummy_progress = torch.tensor(1.0, device=DEVICE, dtype=torch.float64)
    accepted_count = 0

    with torch.no_grad():
        z_curr, log_p_z_curr = prior.sample(1)
        phi_curr, log_det_J_curr = model(z_curr, dummy_progress, enforce_sym=enforce_sym)
        log_q_curr = log_p_z_curr - log_det_J_curr
        S_curr = action_fn(phi_curr)

        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)

            z_prop, log_p_z_prop = prior.sample(current_batch)
            phi_prop, log_det_J_prop = model(z_prop, dummy_progress, enforce_sym=enforce_sym)
            log_q_prop = log_p_z_prop - log_det_J_prop
            S_prop = action_fn(phi_prop)

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
                    curr_S_val = prop_S_val
                    curr_log_q_val = prop_log_q_val
                    phi_curr = phi_prop[j:j + 1]
                    S_curr = S_prop[j:j + 1]
                    log_q_curr = log_q_prop[j:j + 1]
                    accepted_count += 1
                configurations[i + j] = phi_curr.cpu().numpy().reshape(L, L)

    acc_rate = accepted_count / n_samples
    return configurations, acc_rate


# ==========================================
# 3. 严格正则匹配与 ~70% 接受率模型锁定
# ==========================================
def find_and_test_best_model(L):
    script_name = "final_normalizing.py" if L == 14 else f"final_normalizing_L{L}.py"
    if not os.path.exists(script_name):
        raise FileNotFoundError(f"找不到模型脚本: {script_name}")

    spec = importlib.util.spec_from_file_location("nf_module", script_name)
    nf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf_module)
    CONFIG = nf_module.CONFIG
    base_name = CONFIG['base_name']

    acc_pattern = re.compile(r"^acc_(\d+)p(\d+)percent_iter_\d+_" + re.escape(base_name) + r"\.pt$")
    all_files = os.listdir(".")
    pt_files = [f for f in all_files if acc_pattern.match(f)]

    if pt_files:
        closest_file = None
        min_diff = float('inf')
        best_acc = 0.0
        for f in pt_files:
            match = acc_pattern.match(f)
            acc = float(f"{match.group(1)}.{match.group(2)}")
            if abs(acc - 70.0) < min_diff:
                min_diff, closest_file, best_acc = abs(acc - 70.0), f, acc
        return closest_file, best_acc, nf_module, CONFIG

    raw_pattern = re.compile(r"^iter_(\d+)_" + re.escape(base_name) + r"\.pt$")
    raw_files = []
    for f in all_files:
        match = raw_pattern.match(f)
        if match:
            step = int(match.group(1))
            if step <= 30000:
                raw_files.append(f)

    if not raw_files:
        raise FileNotFoundError(f"L={L} 未能找到任何 30000 步以内且严格匹配 base_name '{base_name}' 的存档！")

    print(f"  [🔍 L={L}] 锁定 {len(raw_files)} 个 30000 步以内的 iter_ 存档，开始快速评测...")

    model = nf_module.FlowModel(CONFIG).to(DEVICE).to(torch.float64)
    prior = nf_module.FreeFieldPrior(CONFIG['L'], CONFIG.get('m_sq_prior', 0.6005269985)).to(DEVICE).to(torch.float64)
    action_fn = nf_module.compute_action

    best_raw_file = None
    min_diff = float('inf')
    best_acc = 0.0
    eval_samples = 20000

    for raw_file in raw_files:
        checkpoint = torch.load(raw_file, map_location=DEVICE, weights_only=False)
        model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        model.eval()

        match = raw_pattern.match(raw_file)
        step = int(match.group(1)) if match else 0
        enforce_sym = CONFIG.get('enforce_z2_sym', False) and (step >= CONFIG.get('sym_start_iter', 0))

        _, acc_rate = run_mcmc_chain(model, prior, action_fn, eval_samples,
                                     min(CONFIG.get('batch_size', 10000), eval_samples), enforce_sym, L)
        acc_percent = acc_rate * 100
        diff = abs(acc_percent - 70.0)

        print(f"    -> {raw_file} | 接受率: {acc_percent:.2f}% (离 70% 差 {diff:.2f}%)")

        if diff < min_diff:
            min_diff, best_raw_file, best_acc = diff, raw_file, acc_percent

    acc_str = f"{best_acc:.2f}".replace('.', 'p')
    new_name = f"acc_{acc_str}percent_{best_raw_file}"
    shutil.copy2(best_raw_file, new_name)
    print(f"  [✅ L={L}] 评估完成！最接近 70% 的模型已被存档为: {new_name}\n")

    return new_name, best_acc, nf_module, CONFIG


def generate_nf_mcmc_data(L, n_samples=1000000, batch_size=10000):
    best_file, best_acc, nf_module, CONFIG = find_and_test_best_model(L)
    print(f"  --> 🎯 [L={L}] 启动百万级生成，载入权重: {best_file}")

    model = nf_module.FlowModel(CONFIG).to(DEVICE).to(torch.float64)
    prior = nf_module.FreeFieldPrior(CONFIG['L'], CONFIG.get('m_sq_prior', 0.6005269985)).to(DEVICE).to(torch.float64)
    checkpoint = torch.load(best_file, map_location=DEVICE, weights_only=False)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    model.eval()

    match = re.search(r"iter_(\d+)_", best_file)
    step = int(match.group(1)) if match else 0
    enforce_sym = CONFIG.get('enforce_z2_sym', False) and (step >= CONFIG.get('sym_start_iter', 0))

    print(f"🚀 开始为 L={L} 产出 {n_samples} 真实 MCMC 样本...")
    configurations, final_acc = run_mcmc_chain(model, prior, nf_module.compute_action, n_samples, batch_size,
                                               enforce_sym, L)

    output_filename = f"NF_configs_L{L}_N1000000_acc{best_acc:.1f}.npz"
    np.savez(f"./{output_filename}", configs=configurations)
    print(f"✅ L={L} 生成完毕 (实际采样接受率: {final_acc:.2%})，保存至: {output_filename}\n")
    return configurations


# ==========================================
# 4. 物理可观测量与自相关时间 (加入 Bootstrap 误差评估)
# ==========================================
def calculate_observables(configs):
    configs = np.asarray(configs)
    if configs.ndim == 4:
        if configs.shape[1] == 1:
            configs = configs[:, 0, :, :]
        elif configs.shape[-1] == 1:
            configs = configs[:, :, :, 0]

    _, L, _ = configs.shape
    V = L * L
    d = 2

    M = np.sum(configs, axis=(1, 2))
    chi2_series = ((M - np.mean(M)) ** 2) / V

    phi_right = np.roll(configs, shift=-1, axis=2)
    phi_up = np.roll(configs, shift=-1, axis=1)
    G1_raw = np.sum(configs * phi_right, axis=(1, 2)) / V
    G2_raw = np.sum(configs * phi_up, axis=(1, 2)) / V

    phi_mean = np.mean(configs, axis=0)
    G1_sub = np.sum(phi_mean * np.roll(phi_mean, shift=-1, axis=1)) / V
    G2_sub = np.sum(phi_mean * np.roll(phi_mean, shift=-1, axis=0)) / V

    E_series = ((G1_raw - G1_sub) + (G2_raw - G2_sub)) / d
    Gc0_series = np.mean(configs ** 2, axis=(1, 2))

    return E_series, chi2_series, Gc0_series


def integrated_autocorr_time_with_error(x, c=5.0, n_boot=100, block_size=1000):
    def calc_tau(ts):
        N = len(ts)
        ts = ts - np.mean(ts)
        C0 = np.var(ts, ddof=1)
        if C0 == 0: return 0.5
        tau = 0.5
        for t in range(1, N):
            Ct = np.mean(ts[:N - t] * ts[t:])
            rho_t = Ct / C0
            tau += rho_t
            if t >= c * tau: break
        return max(0.5, tau)

    tau_central = calc_tau(x)

    N = len(x)
    n_blocks = N // block_size
    if n_blocks == 0: return tau_central, 0.0

    x_blocks = x[:n_blocks * block_size].reshape(n_blocks, block_size)
    taus_boot = []

    for _ in range(n_boot):
        idx = np.random.randint(0, n_blocks, size=n_blocks)
        boot_ts = x_blocks[idx].flatten()
        taus_boot.append(calc_tau(boot_ts))

    tau_err = np.std(taus_boot, ddof=1)
    return tau_central, tau_err


# ==========================================
# 5. 主流程与独立的各算法绘图阶段
# ==========================================
def run_scaling_analysis():
    L_list = [6, 8, 10, 12, 14]
    algorithms = ["HMC", "Local", "NF"]
    observables = ["E", "chi2", "Gc0"]

    results = {alg: {obs: {"tau": [], "err": []} for obs in observables} for alg in algorithms}

    for alg in algorithms:
        for L in L_list:
            if alg == "NF":
                nf_pattern = re.compile(rf"^NF_configs_L{L}_N1000000_acc\d+\.\d+\.npz$")
                all_npz = os.listdir(".")
                search_nf = [f for f in all_npz if nf_pattern.match(f)]

                if not search_nf:
                    try:
                        configs = generate_nf_mcmc_data(L, n_samples=1000000)
                    except Exception as e:
                        print(f"  [跳过] L={L} 流程崩溃: {e}")
                        for obs in observables:
                            results[alg][obs]["tau"].append(np.nan)
                            results[alg][obs]["err"].append(np.nan)
                        continue
                else:
                    filename = search_nf[0]
                    print(f"  加载现成数据: {filename}")
                    configs = np.load(filename)['configs']
            else:
                filename = f"{alg}_configs_L{L}_N1280000_DTYPE_double.npz"
                if os.path.exists(filename):
                    print(f"  加载传统算法 [0:1000000]: {filename}")
                    data = np.load(filename)
                    configs = (data['configs'] if 'configs' in data else data['arr_0'])[:1000000]
                else:
                    print(f"  [警告] 缺失文件: {filename}")
                    for obs in observables:
                        results[alg][obs]["tau"].append(np.nan)
                        results[alg][obs]["err"].append(np.nan)
                    continue

            for s_idx, s_data in enumerate(calculate_observables(configs)):
                obs = observables[s_idx]
                tau, err = integrated_autocorr_time_with_error(s_data)
                results[alg][obs]["tau"].append(tau)
                results[alg][obs]["err"].append(err)

            print(f"    --> {alg} L={L} 计算完毕")

    # --- 独立绘图阶段 ---
    plt.style.use(PLOT_CONFIG["style"])
    fit_equations = []
    clean_formatter = FuncFormatter(lambda val, pos: f"{val:g}")

    print("\n🎨 开始渲染分离图表...")

    for alg in algorithms:
        # 每次循环创建一张全新的独立图表
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"])

        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=PLOT_CONFIG["fontsize_label"], rotation=0, labelpad=20)
        ax.set_xlabel(r"$L$", fontsize=PLOT_CONFIG["fontsize_label"], labelpad=5)

        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)

        # 严格的边界裁切
        ax.set_xlim(PLOT_CONFIG["x_limits"])
        ax.set_ylim(PLOT_CONFIG["y_limits"])

        # 严格的数字刻度
        ax.set_xticks(PLOT_CONFIG["x_ticks"])
        ax.set_yticks(PLOT_CONFIG["y_ticks"])
        ax.get_xaxis().set_major_formatter(clean_formatter)
        ax.get_yaxis().set_major_formatter(clean_formatter)

        # 抹杀默认的副刻度（杂乱小竖线）
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        # 核心修改：direction='in' 让刻度朝内，top=True, right=True 让四周都有刻度框
        # 核心修复：通过 which='both' 强制把 Y 轴的主、副刻度统统按回框内！
        ax.tick_params(which='both', direction='in',
                       # top=True,
                       # right=True
                       )
        # 主刻度样式
        ax.tick_params(which='major', labelsize=PLOT_CONFIG["fontsize_tick"], length=6, width=1.2)
        # 副刻度样式（稍微短一点，更美观）
        ax.tick_params(which='minor', length=3, width=1.0)

        color = PLOT_CONFIG["colors"][alg]

        for obs in observables:
            y_data = np.array(results[alg][obs]["tau"])
            y_err = np.array(results[alg][obs]["err"])

            valid_mask = ~np.isnan(y_data)
            if np.sum(valid_mask) == 0: continue

            x_plot = np.array(L_list)[valid_mask]
            y_plot = y_data[valid_mask]
            err_plot = y_err[valid_mask]

            # 纯粹的图例伪造法，不带误差棒的干净 marker
            ax.plot([], [], marker=PLOT_CONFIG["markers"][obs], color=color, linestyle='',
                    markersize=9, fillstyle='none', markeredgewidth=1.8, label=PLOT_CONFIG["labels"][obs])

            # 真实画出数据与误差，但不挂靠图例标签
            ax.errorbar(x_plot, y_plot, yerr=err_plot,
                        marker=PLOT_CONFIG["markers"][obs], color=color, linestyle='',
                        markersize=9, fillstyle='none', markeredgewidth=1.8,
                        capsize=4, elinewidth=1.5)

            # L>=10 的物理极限拟合
            fit_mask = x_plot >= 10
            if np.sum(fit_mask) >= 2:
                x_fit, y_fit = x_plot[fit_mask], y_plot[fit_mask]
                popt, pcov = np.polyfit(np.log10(x_fit), np.log10(y_fit), 1, cov=True)
                z, z_err = popt[0], np.sqrt(pcov[0, 0])
                err_digit = int(round(z_err * 100))

                fit_line_x = np.linspace(min(x_fit) * 0.95, max(x_fit) * 1.05, 50)
                ax.plot(fit_line_x, (10 ** popt[1]) * (fit_line_x ** z), color="red", linestyle="--", linewidth=2.0)

                text_z = f"L^{{{z:.2f}({err_digit})}}" if err_digit > 0 else f"L^{{{z:.2f}}}"
                fit_equations.append(f"[{alg}] {obs}: {text_z}")

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.legend(fontsize=PLOT_CONFIG["fontsize_legend"], loc="upper left", frameon=True, edgecolor='black',
                  handletextpad=0.1)

        plt.tight_layout()

        # 独立导出保存
        output_img = f"CSD_Scaling_Fig7_{alg}.png"
        plt.savefig(output_img, bbox_inches='tight', facecolor='white')

        # 保存完之后关闭画布，释放内存并防止后续绘制重叠
        plt.close(fig)

        print(f"  ✅ 单图已生成: {output_img}")

    print("\n" + "=" * 40)
    print("📊 拟合斜率数据 (供手动排版使用):")
    print("=" * 40)
    for eq in fit_equations:
        print(eq)
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_scaling_analysis()