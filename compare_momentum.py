import os
import glob
import re
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 全局配置
# ==========================================
N_SAMPLES = 1000000
BIN_SIZE = 100
BOOT_TIME = 2000
BATCH_SIZE = 5000  # 根据显存大小可微调

TARGET_ACCS = [50, 55, 60, 65, 70]
MODELS_INFO = {
    'standard_prior_sample': 'final_normalizing',
    'hard_Z2_constraint': 'step0_final_normalizing',
    'soft_Z2_penalty': 'final_normalizing_with_soft_z2'
}
HMC_FILE = "HMC_configs_L14_N1280000_DTYPE_double.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 运行设备: {device}")


# ==========================================
# 2. 核心功能：Bootstrap 误差分析
# ==========================================
def bootstrap_error(data, bin_size=BIN_SIZE, boot_time=BOOT_TIME):
    """通用的分箱重采样误差评估"""
    n_bins = len(data) // bin_size
    if n_bins < 2: return np.mean(data), 0.0

    # 截断多余数据并分箱求均值
    binned_data = data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)

    boot_means = np.zeros(boot_time)
    for i in range(boot_time):
        idx = np.random.randint(0, n_bins, n_bins)
        boot_means[i] = binned_data[idx].mean()

    return boot_means.mean(), boot_means.std()


# ==========================================
# 3. 核心功能：流式 MCMC 采样 (解决内存溢出)
# ==========================================
def generate_and_compute_mcmc(model_name, pt_path, target_n):
    """动态加载模型，并分批运行 MCMC，提取物理量"""
    mod = importlib.import_module(model_name)
    config = mod.CONFIG

    model = mod.FlowModel(config).to(device)
    prior = mod.FreeFieldPrior(L=config['L'], m_sq_prior=0.6005269985).to(device)

    if config.get('double_precision', False):
        model = model.double()
        prior = prior.double()

    checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(clean_dict)
    model.eval()

    # 读取该模型的 enforce_z2_sym 配置
    enforce_sym = config.get('enforce_z2_sym', False)
    dummy_progress = torch.tensor(1.0, device=device)

    M1, M3, M5 = np.zeros(target_n), np.zeros(target_n), np.zeros(target_n)

    curr_s_val, curr_log_q_val = None, None
    curr_phi = None
    accepted_count = 0

    print(f"   ⏳ 开始生成 {target_n} 个样本... (Z2_Sym: {enforce_sym})")

    with torch.no_grad():
        for i in range(0, target_n, BATCH_SIZE):
            current_batch = min(BATCH_SIZE, target_n - i)
            z, log_p_z = prior.sample(current_batch)
            phi, log_det_J = model(z, dummy_progress, enforce_sym=enforce_sym)

            # 计算 Action
            phi_padded = torch.nn.functional.pad(phi, pad=(1, 1, 1, 1), mode='circular')
            adaptive_kernel = mod.laplacian_kernel.to(dtype=phi.dtype, device=phi.device)
            laplacian = torch.nn.functional.conv2d(phi_padded, adaptive_kernel)
            action_density = phi * laplacian + config['m_sq'] * (phi ** 2) + config['lam'] * (phi ** 4)
            prop_s = torch.sum(action_density, dim=(1, 2, 3))

            prop_log_q = log_p_z - log_det_J

            # 转换为 CPU numpy 计算接受率 (极速)
            s_np = prop_s.cpu().numpy()
            log_q_np = prop_log_q.cpu().numpy()
            phi_np = phi.cpu().numpy()
            log_rands = np.log(np.random.rand(current_batch))

            # 初始化马尔可夫链起点
            if curr_s_val is None:
                curr_s_val = s_np[0]
                curr_log_q_val = log_q_np[0]
                curr_phi = phi_np[0]
                M1[0] = curr_phi.mean()
                M3[0] = (curr_phi ** 3).mean()
                M5[0] = (curr_phi ** 5).mean()
                start_idx = 1
            else:
                start_idx = 0

            for j in range(start_idx, current_batch):
                log_acc_ratio = (-s_np[j] - log_q_np[j]) - (-curr_s_val - curr_log_q_val)
                if log_rands[j] < log_acc_ratio:
                    curr_s_val = s_np[j]
                    curr_log_q_val = log_q_np[j]
                    curr_phi = phi_np[j]
                    accepted_count += 1

                # 记录构型的空间均值
                global_idx = i + j
                M1[global_idx] = curr_phi.mean()
                M3[global_idx] = (curr_phi ** 3).mean()
                M5[global_idx] = (curr_phi ** 5).mean()

    actual_acc = accepted_count / (target_n - 1)
    return M1, M3, M5, actual_acc


# ==========================================
# 4. 主流程逻辑
# ==========================================
def main():
    results = {}

    # 4.1 寻找最接近的检查点
    print("\n🔍 正在扫描模型库寻标...")
    for label, mod_name in MODELS_INFO.items():
        results[label] = {}

        mod = importlib.import_module(mod_name)
        base_name = mod.CONFIG['base_name']

        # 获取当前目录下所有 acc_ 开头的 pt 文件
        all_pt_files = glob.glob("acc_*percent_iter_*.pt")
        pt_files = []

        # 🌟 终极修复：使用正则精准抠出文件名中的 base_name 部分，进行绝对等于 (==) 判断
        for f in all_pt_files:
            filename = os.path.basename(f)
            # 解析格式: acc_XXpXXpercent_iter_XXXX_[绝对纯净的base_name].pt
            match = re.match(r'^acc_\d+p\d+percent_iter_\d+_(.+)\.pt$', filename)

            # 只有中间抠出来的名字跟 base_name 一字不差，才收入囊中
            if match and match.group(1) == base_name:
                pt_files.append(f)

        if not pt_files:
            print(f"⚠️ 未找到 {label} 对应的检查点！")
            continue

        acc_dict = {}
        for f in pt_files:
            match_acc = re.search(r'acc_(\d+)p(\d+)percent', f)
            if match_acc:
                acc = float(f"{match_acc.group(1)}.{match_acc.group(2)}")
                acc_dict[acc] = f

        for target in TARGET_ACCS:
            if not acc_dict: break
            closest_acc = min(acc_dict.keys(), key=lambda x: abs(x - target))

            results[label][target] = {
                'file': acc_dict[closest_acc],
                'file_acc': closest_acc
            }
            print(f"   [{label}] 目标 {target}% -> 选中 {closest_acc:.2f}%")

    # 4.2 计算 HMC 基准
    hmc_res = None
    if os.path.exists(HMC_FILE):
        print(f"\n📊 正在计算 HMC 基准数据 (抽取 {N_SAMPLES} 个样本)...")
        data = np.load(HMC_FILE)['configs'][:N_SAMPLES]

        # 🌟 修复：如果数据是 4D [N, 1, L, L] 形状，先去除多余的通道维度
        if data.ndim == 4:
            data = np.squeeze(data, axis=1)  # 变为 [N, L, L]

        # 此时 axis=(1, 2) 完美对应 L, L 两个空间/时间维度
        # 求均值后，hmc_m1 将是一个完美的一维数组 [N, ]，长度为 1000000
        hmc_m1 = data.mean(axis=(1, 2))
        hmc_m3 = (data ** 3).mean(axis=(1, 2))
        hmc_m5 = (data ** 5).mean(axis=(1, 2))

        hmc_res = {
            'phi': bootstrap_error(hmc_m1),
            'phi3': bootstrap_error(hmc_m3),
            'phi5': bootstrap_error(hmc_m5)
        }
        print(f"   ✅ HMC 评估完成！")
    else:
        print(f"⚠️ 未找到 HMC 文件: {HMC_FILE}")

    # 4.3 循环执行模型计算
    print("\n🚀 开始执行流式马尔可夫链评估...")

    table_data = []  # 用于打印表格

    for label, targets in results.items():
        for target_acc, info in targets.items():
            print(f"\n{'=' * 50}")
            print(f"👉 当前模型: {label} | 目标接受率: {target_acc}%")
            print(f"📂 读取权重: {info['file']}")

            M1, M3, M5, actual_acc = generate_and_compute_mcmc(MODELS_INFO[label], info['file'], N_SAMPLES)

            # Bootstrap 分析
            r_m1 = bootstrap_error(M1)
            r_m3 = bootstrap_error(M3)
            r_m5 = bootstrap_error(M5)

            info['res'] = {'phi': r_m1, 'phi3': r_m3, 'phi5': r_m5, 'actual_acc': actual_acc}

            print(f"📈 运行接受率: {actual_acc:.2%} (权重标签: {info['file_acc']:.2f}%)")
            print(f"   <phi>   = {r_m1[0]:.6f} ± {r_m1[1]:.6f}")
            print(f"   <phi^3> = {r_m3[0]:.6f} ± {r_m3[1]:.6f}")
            print(f"   <phi^5> = {r_m5[0]:.6f} ± {r_m5[1]:.6f}")

            table_data.append({
                'Model': label, 'Target': f"{target_acc}%", 'Real_Acc': f"{actual_acc * 100:.1f}%",
                'phi': f"{r_m1[0]:.5f} ± {r_m1[1]:.5f}",
                'phi3': f"{r_m3[0]:.5f} ± {r_m3[1]:.5f}",
                'phi5': f"{r_m5[0]:.5f} ± {r_m5[1]:.5f}"
            })

    # ==========================================
    # 5. 输出 Markdown 表格
    # ==========================================
    print("\n\n" + "=" * 80)
    print(" 📊 Z_2 Symmetry Observables Comparison Table (1M Samples)")
    print("=" * 80)
    header = f"| {'Model':<24} | {'Acc Target':<10} | {'Run Acc':<8} | {'<phi>':<20} | {'<phi^3>':<20} | {'<phi^5>':<20} |"
    print(header)
    print("|" + "-" * 26 + "|" + "-" * 12 + "|" + "-" * 10 + "|" + "-" * 22 + "|" + "-" * 22 + "|" + "-" * 22 + "|")

    # 打印 HMC
    if hmc_res:
        print(
            f"| {'HMC Baseline':<24} | {'N/A':<10} | {'N/A':<8} | {hmc_res['phi'][0]:.5f} ± {hmc_res['phi'][1]:.5f} | {hmc_res['phi3'][0]:.5f} ± {hmc_res['phi3'][1]:.5f} | {hmc_res['phi5'][0]:.5f} ± {hmc_res['phi5'][1]:.5f} |")

    for row in table_data:
        print(
            f"| {row['Model']:<24} | {row['Target']:<10} | {row['Real_Acc']:<8} | {row['phi']:<20} | {row['phi3']:<20} | {row['phi5']:<20} |")
    print("=" * 80 + "\n")

    # ==========================================
    # 6. 生成可视化对比图表 (Errorbar Plots)
    # ==========================================
    print("🎨 正在生成横向对比图表...")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    titles = [r'$\langle \phi \rangle$', r'$\langle \phi^3 \rangle$', r'$\langle \phi^5 \rangle$']
    keys = ['phi', 'phi3', 'phi5']
    colors = {'standard_prior_sample': '#1f77b4', 'hard_Z2_constraint': '#ff7f0e', 'soft_Z2_penalty': '#2ca02c'}
    markers = {'standard_prior_sample': 'o', 'hard_Z2_constraint': 's', 'soft_Z2_penalty': '^'}

    for i, (key, title) in enumerate(zip(keys, titles)):
        ax = axs[i]

        # 绘制 HMC 基准带
        if hmc_res:
            mean, err = hmc_res[key]
            ax.axhline(mean, color='red', linestyle='--', label='HMC Mean')
            ax.axhspan(mean - err, mean + err, color='red', alpha=0.15, label='HMC 1$\sigma$')

        # 遍历三个模型画误差棒
        for label, targets in results.items():
            x_vals, y_vals, y_errs = [], [], []
            for t_acc in TARGET_ACCS:
                if t_acc in targets and 'res' in targets[t_acc]:
                    real_acc = targets[t_acc]['res']['actual_acc'] * 100
                    val, err = targets[t_acc]['res'][key]
                    x_vals.append(real_acc)
                    y_vals.append(val)
                    y_errs.append(err)

            # 对接受率排序连线
            if x_vals:
                sort_idx = np.argsort(x_vals)
                x_vals = np.array(x_vals)[sort_idx]
                y_vals = np.array(y_vals)[sort_idx]
                y_errs = np.array(y_errs)[sort_idx]

                ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt=f'-{markers[label]}',
                            color=colors[label], capsize=4, label=label, alpha=0.8, markersize=6)

        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Acceptance Rate (%)', fontsize=12)
        ax.grid(True, ls="--", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig('Z2_observables_comparison.png')
    print("✅ 图表已保存为: Z2_observables_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()