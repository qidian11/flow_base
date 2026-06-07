import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import os
import glob

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device(
    "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
)
print(f"运行设备: {device}")


def is_valid_checkpoint(filepath):
    try:
        torch.load(filepath, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def auto_find_latest_checkpoint(config):
    layer_str = f"layers_{config['cnn_coupling_layers']}cnn"
    depth_str = f"depth_{config.get('branch_depth', 3)}"

    if config.get('use_multi_kernel', False):
        sizes_str = '_'.join(map(str, config['multi_kernel_sizes']))
        dilations_str = '_'.join(map(str, config.get('multi_kernel_dilations', (1, 1, 1))))
        k_str = f"multi_k_{sizes_str}_dil_{dilations_str}_{depth_str}_{layer_str}"
    else:
        k_str = f"kernel_size_{config['kernel_size']}_{depth_str}_{layer_str}"

    base_pattern = (
        f"bimodal_vacuum_prior_double_precision_*_"
        f"{config['L']}_coupling_layers_{(config['cnn_coupling_layers'])}_"
        f"{k_str}_"
        f"hidden_layers_{config['hidden_layers']}_hidden_channels_{config['hidden_channels']}_"
        f"iterations_*.pt"
    )

    def get_sorted_files(prefix):
        files = glob.glob(prefix + base_pattern)
        return sorted(files, key=os.path.getmtime, reverse=True)

    for file in get_sorted_files("latest_"):
        if is_valid_checkpoint(file): return file
    for file in get_sorted_files("best_"):
        if is_valid_checkpoint(file): return file
    for file in get_sorted_files(""):
        filename = os.path.basename(file)
        if not filename.startswith(("latest_", "best_")) and is_valid_checkpoint(file):
            return file
    return None


# ==========================================
# 1. 物理参数配置 (严格对齐 Z_2 脚本)
# ==========================================
CONFIG = {
    'L': 14,
    'm_sq': -4.0,
    'lam': 5.113,
    'batch_size': 1024,
    'lr': 1e-3,
    'use_scheduler': True,
    'scheduler_min': 1e-5,
    'iterations': 25000,
    'warmup_steps': 100.0,
    'target_acc_ratio': 0.78,

    'cnn_coupling_layers': 6,  # 严格对齐，改为 6 层，依靠网络复用进行 12 次前向
    'kernel_size': 3,
    'hidden_layers': 4,
    'branch_depth': 2,
    'hidden_channels': 16,
    'double_precision': False,
    'use_multi_kernel': True,
    'multi_kernel_sizes': (3,),
    'multi_kernel_dilations': (1,),
}

total_coupling_layers = CONFIG['cnn_coupling_layers']
layer_str = f"layers_{CONFIG['cnn_coupling_layers']}cnn"
depth_str = f"depth_{CONFIG.get('branch_depth', 3)}"

if CONFIG.get('use_multi_kernel', False):
    sizes_str = '_'.join(map(str, CONFIG['multi_kernel_sizes']))
    dilations_str = '_'.join(map(str, CONFIG.get('multi_kernel_dilations', (1, 1, 1))))
    k_str = f"multi_k_{sizes_str}_dil_{dilations_str}_{depth_str}_{layer_str}"
else:
    k_str = f"kernel_size_{CONFIG['kernel_size']}_{depth_str}_{layer_str}"

# 文件名修改为 bimodal_vacuum_prior
save_path = f"best_bimodal_vacuum_prior_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
loss_save_path = f"bimodal_vacuum_prior_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npy"
checkpoint_path = f"latest_bimodal_vacuum_prior_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
phi_ensemble_save_path = f"phi_ensemble_bimodal_vacuum_prior_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npz"

CONFIG['save_path'] = save_path
CONFIG['loss_save_path'] = loss_save_path
CONFIG['checkpoint_path'] = checkpoint_path
CONFIG['phi_ensemble_save_path'] = phi_ensemble_save_path

dtype = torch.float64 if CONFIG.get('double_precision', False) else torch.float32
laplacian_kernel = torch.tensor([[
    [0.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 0.0]
]], device=device, dtype=dtype).unsqueeze(1)


# ==========================================
# 2. 标量场理论的 Action 计算 (对齐 Z_2 提速版)
# ==========================================
def compute_action(phi):
    phi_padded = F.pad(phi, pad=(1, 1, 1, 1), mode='circular')
    laplacian = F.conv2d(phi_padded, laplacian_kernel)
    action_density = phi * laplacian + CONFIG['m_sq'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 掩码生成与卷积上下文网络 (剔除 Z_2 对称性出口)
# ==========================================
def create_checkerboard_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()


class CompiledWeightNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.weight_v = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        with torch.no_grad():
            initial_norm = torch.sqrt(torch.sum(self.weight_v ** 2, dim=(1, 2, 3), keepdim=True))
        self.weight_g = nn.Parameter(initial_norm)

    def forward(self, x):
        v_norm = torch.sqrt(torch.sum(self.weight_v ** 2, dim=(1, 2, 3), keepdim=True))
        w = self.weight_g * (self.weight_v / (v_norm + 1e-8))
        return F.conv2d(x, w, bias=None, stride=1, padding=0, dilation=self.dilation)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, branch_depth=3):
        super().__init__()
        # 动态计算 padding
        pad = kernel_size // 2

        # 🌟 修改：使用动态网络容器堆叠指定数量的 卷积+激活层
        layers = []
        for _ in range(branch_depth):
            layers.append(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=pad, padding_mode='circular'))
            layers.append(nn.LeakyReLU(0.01))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # 🌟 现在的残差连接极其简洁
        return x + self.block(x)


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), branch_depth=3):
        super().__init__()
        self.channels = channels
        assert len(kernel_sizes) == len(dilations), "卷积核数量和空洞率数量必须严格匹配！"
        self.branches = nn.ModuleList()
        self.branch_pads = []

        for k, d in zip(kernel_sizes, dilations):
            assert k % 2 != 0, f"多尺度卷积核必须均为奇数！当前输入了偶数核: {k}"
            pad_size = d * (k - 1) // 2
            self.branch_pads.append(pad_size)

            layers = nn.ModuleList()
            for _ in range(branch_depth):
                layers.append(nn.Conv2d(channels, channels, kernel_size=k, dilation=d, padding=0, bias=False))
                layers.append(nn.LeakyReLU(0.01))
            self.branches.append(layers)

        self.fusion_conv = nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False)

    def _initialize_weights(self):
        nn.init.normal_(self.fusion_conv.weight, mean=0.0, std=0.01)

    def forward(self, x):
        outs = []
        for branch, pad_size in zip(self.branches, self.branch_pads):
            out = x
            pad_tuple = (pad_size, pad_size, pad_size, pad_size)
            for i in range(0, len(branch), 2):
                conv_layer = branch[i]
                act_layer = branch[i + 1]
                out = F.pad(out, pad=pad_tuple, mode='circular')
                out = act_layer(conv_layer(out))
            outs.append(out)

        fused = torch.cat(outs, dim=1)
        out = self.fusion_conv(fused)
        return x + out


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4, kernel_size=3, use_multi_kernel=False,
                 multi_kernel_sizes=(3, 5, 7), multi_kernel_dilations=(1, 1, 1), branch_depth=3):
        super().__init__()

        # 🌟 构造独立的子网络，向 bimodal_z_2 的 full_capacity_net 严格看齐
        def build_independent_net():
            layers = []
            # 入口层：对齐 circular padding 和 bias=True，确保感受野和参数量完全一致
            layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=True))
            layers.append(nn.LeakyReLU(0.01))

            for _ in range(num_hidden_layers):
                if use_multi_kernel:
                    layers.append(MultiScaleResBlock(hidden_channels, kernel_sizes=multi_kernel_sizes,
                                                     dilations=multi_kernel_dilations, branch_depth=branch_depth))
                else:
                    layers.append(ResBlock(hidden_channels, kernel_size=kernel_size, branch_depth=branch_depth))

            # 🌟 出口层：移除冗余的 1x1 隐藏过渡层，直接降维输出 1 个通道 (s 或 t)
            layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True))
            return nn.Sequential(*layers)

        # 拆分为完全独立的 s 网络和 t 网络
        self.s_net = build_independent_net()
        self.t_net = build_independent_net()

        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化 s 网络：确保初始尺度收缩趋近于 1 (即 s=0)
        nn.init.normal_(self.s_net[0].weight, mean=0, std=0.1)
        nn.init.zeros_(self.s_net[-1].weight)
        nn.init.zeros_(self.s_net[-1].bias)

        # 初始化 t 网络：确保初始平移趋近于 0
        nn.init.normal_(self.t_net[0].weight, mean=0, std=0.1)
        nn.init.zeros_(self.t_net[-1].weight)
        nn.init.zeros_(self.t_net[-1].bias)

    def forward(self, x):
        # 分别进行独立前向计算
        s_out = self.s_net(x)
        t_out = self.t_net(x)

        # 拼接为 (B, 2, H, W) 以便无缝接入外部 FlowModel 原有的拆分逻辑
        return torch.cat([s_out, t_out], dim=1)


# ==========================================
# 4. 双峰拓扑混合先验 (Gaussian Mixture Prior)
# ==========================================
class BimodalVacuumPrior(nn.Module):
    def __init__(self, L, m_eff_sq, v):
        super().__init__()
        self.L = L
        self.V = L * L
        self.register_buffer('v', torch.tensor(v))  # 保存背景场强度

        p = torch.arange(L) * 2.0 * math.pi / L
        P1, P2 = torch.meshgrid(p, p, indexing='ij')
        K = m_eff_sq + 4.0 * torch.sin(P1 / 2.0) ** 2 + 4.0 * torch.sin(P2 / 2.0) ** 2

        self.register_buffer('sqrt_2K', torch.sqrt(2.0 * K).view(1, 1, L, L))
        self.register_buffer('log_det_factor', 0.5 * torch.sum(torch.log(2.0 * K)))
        self.register_buffer('const_factor', torch.tensor(0.5 * self.V * math.log(2.0 * math.pi)))

    def sample(self, batch_size):
        # 1. 采样自由场量子涨落 (基于 m_eff_sq)
        eta = torch.randn(batch_size, 1, self.L, self.L, device=self.sqrt_2K.device, dtype=self.sqrt_2K.dtype)
        eta_k = torch.fft.fftn(eta, dim=(-2, -1), norm="ortho")
        phi_k = eta_k / self.sqrt_2K
        phi_fluct = torch.fft.ifftn(phi_k, dim=(-2, -1), norm="ortho").real

        # 2. 抛硬币决定拓扑扇区 (生成 +v 或 -v 的背景场)
        # torch.randint(0, 2) 生成 0 或 1，映射为 -1 或 1
        signs = torch.randint(0, 2, (batch_size, 1, 1, 1), device=self.sqrt_2K.device, dtype=self.sqrt_2K.dtype) * 2 - 1
        phi_bg = signs * self.v

        # 3. 组装最终采样场
        phi_sample = phi_bg + phi_fluct

        # 4. 计算高斯混合分布的精确 Log Prob (核心数值保护区)
        # 必须分别计算当前样本属于 +v 峰和 -v 峰的独立对数概率
        log_p_plus = self._compute_single_log_prob(phi_sample - self.v)
        log_p_minus = self._compute_single_log_prob(phi_sample + self.v)

        # 严格的 log( 0.5 * exp(p_plus) + 0.5 * exp(p_minus) )
        # = logsumexp([p_plus, p_minus]) - log(2)
        log_p_stacked = torch.stack([log_p_plus, log_p_minus], dim=1)
        log_p_mix = torch.logsumexp(log_p_stacked, dim=1) - math.log(2.0)

        return phi_sample, log_p_mix

    def _compute_single_log_prob(self, phi_shifted):
        """计算场偏离给定真空后的纯高斯涨落对数概率"""
        phi_shifted_k = torch.fft.fftn(phi_shifted, dim=(-2, -1), norm="ortho")
        eta_reconstructed_k = phi_shifted_k * self.sqrt_2K
        eta_reconstructed = torch.fft.ifftn(eta_reconstructed_k, dim=(-2, -1), norm="ortho").real
        return -0.5 * torch.sum(eta_reconstructed ** 2, dim=(1, 2, 3)) - self.const_factor + self.log_det_factor


# ==========================================
# [新增] 泛函极值求解器：计算双峰拓扑先验 (Bimodal Vacuum Prior)
# ==========================================
def solve_bimodal_gap_equations(L, m_sq_bare, lam, device, tol=1e-7, max_iter=1000):
    """
    严格求解破缺相 (m^2 < 0) 的耦合能隙方程组：
    1. 希格斯有效质量 m_eff^2 = -2m^2 - \lambda \Sigma
    2. 量子修正真空期望值 v = \sqrt{ -6m^2/\lambda - 3\Sigma }
    """
    dtype = torch.float64
    p = torch.arange(L, device=device, dtype=dtype) * 2.0 * math.pi / L
    P1, P2 = torch.meshgrid(p, p, indexing='ij')
    K_hat = 4.0 * torch.sin(P1 / 2.0) ** 2 + 4.0 * torch.sin(P2 / 2.0) ** 2

    if m_sq_bare >= 0:
        raise ValueError("此求解器专为破缺相 (m^2 < 0) 设计。对称相请使用单峰穿衣质量。")

    # 初始猜测：以经典树图级别的真空和希格斯质量为起点
    m_eff_sq = torch.tensor(-2.0 * m_sq_bare, device=device, dtype=dtype)
    sigma = torch.tensor(0.0, device=device, dtype=dtype)

    for i in range(max_iter):
        # 1. 计算单圈蝌蚪图积分 (涨落的方差)
        sigma_new = torch.mean(1.0 / (K_hat + m_eff_sq))

        # 2. 更新希格斯有效质量
        m_eff_sq_new = -2.0 * m_sq_bare - lam * sigma_new
        if m_eff_sq_new <= 0:
            m_eff_sq_new = torch.tensor(1e-4, device=device, dtype=dtype)

        # 3. 更新真空期望值 (VEV) 的平方
        v_sq_new = (-6.0 * m_sq_bare / lam) - 3.0 * sigma_new
        if v_sq_new <= 0:
            v_sq_new = torch.tensor(1e-4, device=device, dtype=dtype)

        # 检查收敛
        if torch.abs(m_eff_sq_new - m_eff_sq) < tol and torch.abs(sigma_new - sigma) < tol:
            vev = math.sqrt(v_sq_new.item())
            print(f"🌟 双峰能隙方程收敛! (迭代 {i} 次)")
            print(f"   ├─ 希格斯有效质量^2: {m_eff_sq_new.item():.4f}")
            print(f"   └─ 量子修正真空期望值 ±v: {vev:.4f}")
            return m_eff_sq_new.item(), vev

        m_eff_sq = m_eff_sq_new
        sigma = sigma_new

    vev = math.sqrt(max(1e-4, ((-6.0 * m_sq_bare / lam) - 3.0 * sigma.item())))
    print(f"⚠️ 双峰能隙方程未完全收敛，返回最后近似值 | 质量^2: {m_eff_sq.item():.4f}, v: {vev:.4f}")
    return m_eff_sq.item(), vev


# ==========================================
# 5. 流模型 (包含防爆盾和双循环重用)
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config['L']
        self.cnn_layers = config['cnn_coupling_layers']
        self.total_layers = self.cnn_layers
        self.register_buffer('base_mask', create_checkerboard_mask(self.L))
        self.context_nets = nn.ModuleList()
        dtype = torch.float64 if config.get('double_precision', False) else torch.float32

        # 防爆盾边界
        self.register_buffer('s_bounds', torch.tensor([-0.5, 4.0], dtype=dtype))
        self.register_buffer('t_bounds', torch.tensor([-15.0, 15.0], dtype=dtype))

        for _ in range(self.cnn_layers):
            self.context_nets.append(ConvContextNet(
                hidden_channels=config['hidden_channels'], num_hidden_layers=config['hidden_layers'],
                kernel_size=config['kernel_size'], use_multi_kernel=config.get('use_multi_kernel', False),
                multi_kernel_sizes=config.get('multi_kernel_sizes', (3, 3)),
                multi_kernel_dilations=config.get('multi_kernel_dilations', (1, 2)),
                branch_depth=config.get('branch_depth', 3)
            ))

        if config.get('use_multi_kernel', False):
            arch_info = f"多尺度: {config.get('multi_kernel_sizes')} | 空洞率: {config.get('multi_kernel_dilations')}"
        else:
            arch_info = f"单核: {config['kernel_size']}"

        print("=" * 70)
        print(f"🌟 对齐版 Prior_CNN 物理流模型初始化完毕！")
        print(f"👉 总耦合层数: {self.total_layers} 层 (双步复用, 特征通道数: {config['hidden_channels']})")
        print(f"   │")
        print(f"   ├─ [结构对齐] 网络不具备内部 Z_2 结构，全靠数据驱动学习")
        print(f"      ├─ 内部隐藏层 (ResBlocks): {config['hidden_layers']} 层 / 耦合层")
        print(f"      └─ 卷积网络配置: {arch_info} | 分支深度: {config.get('branch_depth', 3)}")
        print("=" * 70)

    def step_warmup(self, progress):
        """只负责放宽边界，防爆盾的开关由外部的 .train() 和 .eval() 决定"""
        if progress < 1.0:
            self.s_bounds[0].fill_(-0.5 - 1.5 * progress)
            self.s_bounds[1].fill_(4.0 + 4.0 * progress)
            self.t_bounds[0].fill_(-15.0 - 15.0 * progress)
            self.t_bounds[1].fill_(15.0 + 15.0 * progress)
        else:
            self.s_bounds[0].fill_(-100.0)
            self.s_bounds[1].fill_(100.0)
            self.t_bounds[0].fill_(-100.0)
            self.t_bounds[1].fill_(100.0)

    def forward(self, z, progress=None):
        phi = z
        log_det_jacobian = 0

        # 对齐 Z_2 脚本的参数复用逻辑：每个网络跑两次 (奇数格/偶数格)
        for net in self.context_nets:
            for step in range(2):
                current_mask = self.base_mask if step == 0 else (1.0 - self.base_mask)
                phi_frozen = current_mask * phi

                # 拿到没有任何 Z_2 保护的标准 s 和 t
                st_out = net(phi_frozen)
                s_out, t_out = st_out[:, 0:1, :, :], st_out[:, 1:2, :, :]

                if self.training and progress is not None:
                    is_warmup = (progress < 1.0).view(1, 1, 1, 1)

                    s_clamped = asymmetric_soft_clamp(s_out, self.s_bounds[0], self.s_bounds[1])
                    t_clamped = torch.clamp(t_out, self.t_bounds[0], self.t_bounds[1])

                    s_out = torch.where(is_warmup, s_clamped, s_out)
                    t_out = torch.where(is_warmup, t_clamped, t_out)

                update_mask = 1.0 - current_mask

                # 🌟 统一仿射公式为逆向方程： y = (x - t) * exp(-s)
                phi = phi_frozen + update_mask * ((phi - t_out) * torch.exp(-s_out))
                log_det_jacobian += torch.sum(update_mask * (-s_out), dim=(1, 2, 3))

        return phi, log_det_jacobian


def asymmetric_soft_clamp(x, min_val, max_val):
    pos_val = max_val * torch.tanh(x / max_val)
    neg_val = abs(min_val) * torch.tanh(x / abs(min_val))
    return torch.where(x >= 0, pos_val, neg_val)


# ==========================================
# 6. 一体化评估核心
# ==========================================
def run_mcmc_evaluation(model, prior, total_n=10000, batch_size=1024):
    model.eval()
    total_n = (total_n // batch_size) * batch_size

    dtype = torch.float64 if CONFIG.get('double_precision', False) else torch.float32
    all_s = torch.empty(total_n, dtype=dtype, device=device)
    all_log_qs = torch.empty(total_n, dtype=dtype, device=device)

    dummy_progress = torch.tensor(1.0, device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - i)
            z, log_p_z = prior.sample(current_batch)
            phi, log_det_J = model(z, dummy_progress)

            all_s[i:i + current_batch] = compute_action(phi)
            all_log_qs[i:i + current_batch] = log_p_z - log_det_J

    accepted_count = 0
    curr_s = all_s[0]
    curr_log_q = all_log_qs[0]

    log_rands = torch.log(torch.rand(total_n, device=device, dtype=dtype))

    for i in range(1, total_n):
        prop_s = all_s[i]
        prop_log_q = all_log_qs[i]

        log_acc_ratio = (-prop_s - prop_log_q) - (-curr_s - curr_log_q)

        if log_rands[i] < log_acc_ratio:
            curr_s = prop_s
            curr_log_q = prop_log_q
            accepted_count += 1

    model.train()
    return accepted_count / (total_n - 1)


# ==========================================
# 7. 训练主循环 (包含早停机制)
# ==========================================
def train():
    model = FlowModel(CONFIG).to(device)

    # ====== [核心修改] 注入拓扑双峰变分先验 ======
    print("\n" + "=" * 70)
    print("🛠️ 正在求解耦合能隙方程组，计算拓扑混合先验 (Bimodal Vacuum Prior)...")
    m_eff_sq, vev = solve_bimodal_gap_equations(CONFIG['L'], CONFIG['m_sq'], CONFIG['lam'], device)
    print("=" * 70 + "\n")

    # 使用解出的希格斯有效质量和真空期望值初始化双峰先验
    prior = BimodalVacuumPrior(L=CONFIG['L'], m_eff_sq=m_eff_sq, v=vev).to(device)

    if CONFIG['double_precision']:
        model = model.double()
        prior = prior.double()

    eps_val = 1e-15 if CONFIG.get('double_precision', False) else 1e-8
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], eps=eps_val)

    history_loss = []
    best_loss = float('inf')
    ema_loss = None
    start_iteration = 1

    target_milestones = sorted([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75], reverse=True)
    achieved_milestones = set()

    old_checkpoint_path = auto_find_latest_checkpoint(CONFIG)
    if old_checkpoint_path and os.path.exists(old_checkpoint_path):
        print(f"从 {old_checkpoint_path} 恢复训练...")
        checkpoint = torch.load(old_checkpoint_path, map_location=device)

        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(clean_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if CONFIG.get('double_precision', False):
            for group in optimizer.param_groups:
                group['eps'] = 1e-15
        if CONFIG.get('double_precision', False):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point(): state[k] = v.double()

        start_iteration = checkpoint['iteration'] + 1
        history_loss = checkpoint.get('history_loss', [])
        ema_loss = checkpoint.get('ema_loss', None)
        best_loss = checkpoint.get('best_loss', min(history_loss) if history_loss else float('inf'))
        achieved_milestones = checkpoint.get('achieved_milestones', set())

    scheduler = None
    if CONFIG.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['iterations'],
            eta_min=CONFIG['scheduler_min'],
            last_epoch=start_iteration - 1 if start_iteration > 1 else -1
        )

    if hasattr(torch, 'compile') and device.type == 'cuda':
        model.train()
        compiled_model = torch.compile(model)
    else:
        compiled_model = model

    model.train()
    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()
        z, log_p_z = prior.sample(CONFIG['batch_size'])

        warmup_steps = CONFIG['warmup_steps']
        progress_val = min(iteration / warmup_steps, 1.0)
        model.step_warmup(progress_val)
        progress_tensor = torch.tensor(progress_val, device=device, dtype=dtype)

        phi, log_det_J = compiled_model(z, progress_tensor)
        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("⚠️ 捕获到 NaN 梯度！跳过本次更新。")
            optimizer.zero_grad()
            continue
        else:
            optimizer.step()

        if scheduler: scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        if iteration % 100 == 0:
            print(
                f"迭代 {iteration:6d}/{CONFIG['iterations']} | 瞬时 Loss: {loss_val:.4f} | 平滑 Loss: {ema_loss:.4f} | Best: {best_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss_val, 'best_loss': best_loss,
                        'history_loss': history_loss, 'ema_loss': ema_loss,
                        'achieved_milestones': achieved_milestones},
                       checkpoint_path)

        if ema_loss < best_loss and iteration >= 10000:
            best_loss = ema_loss
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss, 'ema_loss': ema_loss,
                        'achieved_milestones': achieved_milestones},
                       save_path)

        if iteration >= 10000 and iteration % 2000 == 0:
            print(f"\n{'=' * 50}")
            print(f"🚀 [迭代 {iteration}] 触发 MCMC 在线验证 (样本量: 10000)...")
            acc_rate = run_mcmc_evaluation(compiled_model, prior, total_n=10000, batch_size=CONFIG['batch_size'], )
            print(f"📊 当前物理接受率: {acc_rate:.2%}")
            print(f"{'=' * 50}\n")

            for m in target_milestones:
                if acc_rate >= m and m not in achieved_milestones:
                    print(f"⭐ 达成里程碑！接受率首次突破 {m:.0%} (当前 {acc_rate:.2%})，正在保存专属模型...")

                    base_name = checkpoint_path.replace('latest_', '')
                    milestone_path = f"acc_{int(m * 100)}percent_{base_name}"

                    for lm in target_milestones:
                        if lm <= m:
                            achieved_milestones.add(lm)

                    torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                                'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss,
                                'ema_loss': ema_loss,
                                'achieved_milestones': achieved_milestones},
                               milestone_path)
                    break

            if acc_rate >= CONFIG['target_acc_ratio']:
                torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'loss': loss_val,
                            'best_loss': best_loss,
                            'history_loss': history_loss,
                            'ema_loss': ema_loss, }, 'mcmc_success_' + checkpoint_path)
                print(f"🎉 成功达标！接受率已达到 {acc_rate:.2%} (>= {CONFIG['target_acc_ratio']:.0%})，提前结束训练阶段！")
                break

    print("✅ 训练流水线结束！")
    return model


if __name__ == "__main__":
    if CONFIG.get('double_precision', False): torch.set_default_dtype(torch.float64)

    final_model = train()