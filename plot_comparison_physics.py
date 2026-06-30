import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from final_normalizing import CONFIG as ml_config

L = 14
MAX_N = 1000000
BIN_SIZE = 100
BOOT_TIME = 2000

# ==========================================
# 1. 配置文件路径
# ==========================================
FILES = {
    'HMC': 'HMC_configs_L14_N1280000_DTYPE_double.npz',
    'Local': 'Local_configs_L14_N1280000_DTYPE_double.npz',
    'ML': ml_config['phi_ensemble_save_path']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前计算设备: {device}")


# ==========================================
# 2. 计算观测量并执行 100万 强行截断
# ==========================================
def compute_observables(file_path, name):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return None

    print(f"\n🚀 正在处理 [{name}] | 数据源: {file_path}")
    data = np.load(file_path)
    configs_np = data['configs']

    if len(configs_np) > MAX_N:
        configs_np = configs_np[:MAX_N]
        print(f"✂️ 已触发超量截断: 截取前 {MAX_N} 个构型")
    elif len(configs_np) < MAX_N:
        print(f"⚠️ 警告: 该文件构型少于 {MAX_N} ({len(configs_np)})")

    configs = torch.tensor(configs_np, dtype=torch.float64, device=device)
    if configs.ndim == 4:
        configs = configs.squeeze(1)

    N = configs.shape[0]
    n_bins = N // BIN_SIZE
    configs = configs[:n_bins * BIN_SIZE]
    print(f"📊 用于计算的有效样本数: {len(configs)} | Bins 数量: {n_bins}")

    configs_k = torch.fft.fft2(configs)
    power_spec = torch.abs(configs_k) ** 2
    C_i = torch.fft.ifft2(power_spec).real / (L * L)
    M_i = configs.mean(dim=(1, 2))

    C_bin = C_i.view(n_bins, BIN_SIZE, L, L).mean(dim=1)
    M_bin = M_i.view(n_bins, BIN_SIZE).mean(dim=1)

    boot_G_tilde = torch.zeros((BOOT_TIME, L), dtype=torch.float64, device=device)
    boot_m_eff = torch.zeros((BOOT_TIME, L), dtype=torch.float64, device=device)

    for i in range(BOOT_TIME):
        idx = torch.randint(0, n_bins, (n_bins,), device=device)
        C_boot = C_bin[idx].mean(dim=0)
        M_boot = M_bin[idx].mean(dim=0)

        G_c = C_boot - M_boot ** 2
        G_tilde = G_c.sum(dim=1)
        boot_G_tilde[i] = G_tilde

        G_tilde_plus = torch.roll(G_tilde, shifts=-1, dims=0)
        G_tilde_minus = torch.roll(G_tilde, shifts=1, dims=0)
        ratio = (G_tilde_plus + G_tilde_minus) / (2 * G_tilde + 1e-12)
        ratio = torch.clamp(ratio, min=1.0000001)
        boot_m_eff[i] = torch.acosh(ratio)

    return {
        'G_mean': boot_G_tilde.mean(dim=0).cpu().numpy(),
        'G_err': boot_G_tilde.std(dim=0).cpu().numpy(),
        'm_mean': boot_m_eff.mean(dim=0).cpu().numpy(),
        'm_err': boot_m_eff.std(dim=0).cpu().numpy(),
    }


results = {}
for name, path in FILES.items():
    res = compute_observables(path, name)
    if res is not None:
        results[name] = res

# ==========================================
# 3. 完美复刻 Mathematica 画风 (纯白底色 + 内置 LaTeX 引擎 + 双格式保存)
# ==========================================
if len(results) > 0:
    bg_color = 'white'

    # 关闭外部 LaTeX，改用 Matplotlib 自带的 MathText 模拟 Computer Modern 字体
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color
    })

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

    t_axis = np.arange(L)

    styles = {
        'HMC': {'color': 'black', 'marker': 'o', 'label': 'HMC'},
        'Local': {'color': '#e600e6', 'marker': 's', 'label': 'Local'},
        'ML': {'color': 'darkblue', 'marker': 'D', 'label': 'ML'}
    }

    for idx, (name, data) in enumerate(results.items()):
        style = styles[name]

        # 左图：零动量连通格林函数 G_c(0, t)
        axs[0].errorbar(t_axis, data['G_mean'], yerr=data['G_err'],
                        fmt=f'-{style["marker"]}', color=style['color'],
                        markerfacecolor='none', markeredgewidth=1.2,
                        linewidth=1.2, capsize=3, label=style['label'])

        # 右图：有效极点质量 m_p^{eff}
        slice_idx = slice(1, -1)
        axs[1].errorbar(t_axis[slice_idx], data['m_mean'][slice_idx], yerr=data['m_err'][slice_idx],
                        fmt=f'-{style["marker"]}', color=style['color'],
                        markerfacecolor='none', markeredgewidth=1.2,
                        linewidth=1.2, capsize=3, label=style['label'])

    # --- 统一坐标轴精细排版 ---
    for i in range(2):
        axs[i].tick_params(direction='in', top=True, right=True, labelsize=14, length=5)
        for spine in axs[i].spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

        axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[i].text(1.02, -0.05, r'$t$', transform=axs[i].transAxes, fontsize=18)

        axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3,
                      frameon=False, fontsize=15, handlelength=1.5, handletextpad=0.4)

    # 左图专有排版
    axs[0].set_yscale('log')
    axs[0].set_xticks(t_axis)
    axs[0].text(-0.05, 1.05, r'$\tilde{G}_c(0,t)$', transform=axs[0].transAxes, fontsize=18)

    # 右图专有排版
    axs[1].set_xticks(t_axis[1:-1])
    axs[1].text(-0.05, 1.05, r'$m_p^{\rm eff}$', transform=axs[1].transAxes, fontsize=18)

    y_min, y_max = axs[1].get_ylim()
    axs[1].set_ylim(y_min, y_max + (y_max - y_min) * 0.15)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, top=0.85)

    # ==========================================
    # 🌟 修改点：同时保存 PDF 和 高清 PNG，并强制锁定纯白底色
    # ==========================================
    save_pdf = 'physics_observables_comparison.pdf'
    save_png = 'physics_observables_comparison.png'

    # transparent=False 确保背景一定是白色，不会被阅读器自带的底色干扰
    plt.savefig(save_pdf, bbox_inches='tight', facecolor='white', transparent=False)
    # dpi=300 保证 PNG 放大后依然清晰如矢量图
    plt.savefig(save_png, bbox_inches='tight', facecolor='white', transparent=False, dpi=300)

    print(f"\n🎉 纯白底色版的对比图已导出：")
    print(f"   📄 矢量图 (论文用) -> {save_pdf}")
    print(f"   🖼️ 位图 (预览/展示用) -> {save_png}")
    plt.show()