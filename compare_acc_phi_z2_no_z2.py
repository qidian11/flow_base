import re
import matplotlib.pyplot as plt


def parse_log(file_path):
    steps, acc_rates, phi_means, phi_errs = [], [], [], []
    current_step, current_acc = None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 提取步数
            step_match = re.search(r'\[迭代\s+(\d+)\]\s+触发', line)
            if step_match:
                current_step = int(step_match.group(1))
                continue

            # 2. 提取接受率
            acc_match = re.search(r'当前物理接受率:\s+([\d.]+)%', line)
            if acc_match and current_step is not None:
                current_acc = float(acc_match.group(1))
                continue

            # 3. 提取 phi^1 期望与误差
            phi_match = re.search(r'phi\^1:\s+([-\d.]+)\s+±\s+([-\d.]+)', line)
            if phi_match and current_step is not None and current_acc is not None:
                steps.append(current_step)
                acc_rates.append(current_acc)
                phi_means.append(float(phi_match.group(1)))
                phi_errs.append(float(phi_match.group(2)))

                current_step, current_acc = None, None

    return steps, acc_rates, phi_means, phi_errs


def filter_data(steps, acc, mean, err, start_step):
    """过滤指定步数之后的数据，确保 Y 轴可以完美自适应放大"""
    f_steps, f_acc, f_mean, f_err = [], [], [], []
    for s, a, m, e in zip(steps, acc, mean, err):
        if s >= start_step:
            f_steps.append(s)
            f_acc.append(a)
            f_mean.append(m)
            f_err.append(e)
    return f_steps, f_acc, f_mean, f_err


def draw_comparison_plot(s1, a1, m1, e1, s2, a2, m2, e2, title_suffix, save_name):
    """独立的绘图函数，方便复用"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ----- 子图 1: 接受率 -----
    ax1.plot(s1, a1, marker='o', markersize=4, linestyle='-', alpha=0.8, label='prior_cnn_macro_z2')
    ax1.plot(s2, a2, marker='s', markersize=4, linestyle='-', alpha=0.8, label='aligned_prior_cnn')
    ax1.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax1.set_title(f'MCMC Acceptance Rate {title_suffix}', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ----- 子图 2: phi^1 期望 -----
    ax2.errorbar(s1, m1, yerr=e1, fmt='-o', markersize=4, capsize=3, alpha=0.8, label='prior_cnn_macro_z2')
    ax2.errorbar(s2, m2, yerr=e2, fmt='-s', markersize=4, capsize=3, alpha=0.8, label='aligned_prior_cnn')

    # 零点基准线
    ax2.axhline(0, color='black', linestyle=':', alpha=0.8, label='Theoretical Zero')

    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel(r'$\langle \phi \rangle$', fontsize=12)
    ax2.set_title(r'$\phi^1$ Expectation ' + title_suffix, fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()  # 释放内存
    print(f"✅ 图表已保存: {save_name}")


# ==========================================
# 主程序逻辑
# ==========================================
file1 = "prior_cnn_macro_z2 (2).txt"
file2 = "aligned_prior_cnn (3).txt"

# 解析原始数据
steps1, acc1, mean1, err1 = parse_log(file1)
steps2, acc2, mean2, err2 = parse_log(file2)

# 1. 绘制全局图 (0步开始)
draw_comparison_plot(
    steps1, acc1, mean1, err1,
    steps2, acc2, mean2, err2,
    title_suffix="(Global View)",
    save_name="mcmc_comparison_full.png"
)

# 2. 截取并绘制 18000 步之后的局部放大图
ZOOM_START = 18000
z_steps1, z_acc1, z_mean1, z_err1 = filter_data(steps1, acc1, mean1, err1, ZOOM_START)
z_steps2, z_acc2, z_mean2, z_err2 = filter_data(steps2, acc2, mean2, err2, ZOOM_START)

draw_comparison_plot(
    z_steps1, z_acc1, z_mean1, z_err1,
    z_steps2, z_acc2, z_mean2, z_err2,
    title_suffix=f"(Zoomed > {ZOOM_START} steps)",
    save_name="mcmc_comparison_zoomed.png"
)