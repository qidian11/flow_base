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
    # 重复一遍感受野字符串的生成逻辑
    if config.get('s_use_multi_kernel', True):
        s_k_str = '_'.join(map(str, config.get('s_kernel_sizes', (3, 3))))
        s_dil_str = '_'.join(map(str, config.get('s_dilations', (1, 2))))
        s_rf_str = f"sk_{s_k_str}_sdil_{s_dil_str}"
    else:
        s_rf_str = f"sk_{config.get('s_kernel_sizes', (3,))[0]}_sdil_1"

    t_rf_str = f"tk_{config.get('t_kernel_size', 3)}"

    # 🌟 修改匹配模式
    base_pattern = (
        f"{config.get('type', 'asymmetric_prior_cnn')}_double_precision_*_"
        f"{config['L']}_coupling_layers_{config['cnn_coupling_layers']}_depth_*_"
        f"s_ch_{config.get('s_channels', 128)}_t_ch_{config.get('t_channels', 32)}_"
        f"s_ly_{config.get('s_layers', 4)}_t_ly_{config.get('t_layers', 2)}_"
        f"{s_rf_str}_{t_rf_str}_"  # 匹配感受野
        f"iterations_*.pt"
    )

    def get_sorted_files(prefix):
        files = glob.glob(prefix + base_pattern)
        return sorted(files, key=os.path.getmtime, reverse=True)

    for file in get_sorted_files("latest_"):
        if is_valid_checkpoint(file): return file
    for file in get_sorted_files("best_"):
        if is_valid_checkpoint(file): return file
    # 🌟 新增：让 glob 支持搜索 iter_ 开头的文件，并按时间倒序拿最新的
    for file in get_sorted_files("iter_*_"):
        if is_valid_checkpoint(file): return file

    for file in get_sorted_files(""):
        filename = os.path.basename(file)
        # 排除其他前缀，严格匹配普通 .pt 权重
        if not filename.startswith(("latest_", "best_", "iter_")) and is_valid_checkpoint(file):
            return file
    return None


# ==========================================
# 1. 物理参数配置 (严格对齐 Z_2 脚本)
# ==========================================
CONFIG = {
    'type': 'asymmetric_prior_cnn',
    'L': 14,
    'm_sq': -4.0,
    'lam': 5.113,
    'batch_size': 512,
    'lr': 1e-3,
    'use_scheduler': True,
    'scheduler_min': 1e-5,
    'iterations': 60000,          # 🌟 修改：总步数改为 35000 (25000 + 10000)
    'scheduler_steps': 35000,     # 🌟 新增：前多少步使用调度器
    # 'warmup_steps': 18000,      # 废弃
    'target_acc_ratio': 0.78,

    # 🌟 新增：显式 Z_2 对称性强制开关
    'enforce_z2_sym': False,        # 是否开启严格对称
    'sym_start_iter': 30000,       # 在第几步之后开启

# 🌟 核心：非对称参数配置
    's_channels': 96,   # s 网络的通道数
    't_channels': 32,    # t 网络的通道数
    's_layers': 3,       # s 网络的隐藏层数
    't_layers': 2,       # t 网络的隐藏层数

    'cnn_coupling_layers': 6,  # 严格对齐，改为 6 层，依靠网络复用进行 12 次前向
    'branch_depth': 2,
    'double_precision': False,
    # 🌟 2. 感受野 (Receptive Field) 非对称配置
    's_use_multi_kernel': True,
    's_kernel_sizes': (3, 3),   # s 网络的卷积核组合
    's_dilations': (1, 2),      # s 网络的空洞率组合 (负责捕捉长程关联)
    't_kernel_size': 3,         # t 网络固定单核小感受野
}

total_coupling_layers = CONFIG['cnn_coupling_layers']
layer_str = f"layers_{CONFIG['cnn_coupling_layers']}cnn"
depth_str = f"depth_{CONFIG.get('branch_depth', 3)}"

# ==========================================
# 动态生成文件名核心
# ==========================================
# 提取 s 网络的感受野特征
if CONFIG.get('s_use_multi_kernel', True):
    s_k_str = '_'.join(map(str, CONFIG.get('s_kernel_sizes', (3, 3))))
    s_dil_str = '_'.join(map(str, CONFIG.get('s_dilations', (1, 2))))
    s_rf_str = f"sk_{s_k_str}_sdil_{s_dil_str}"
else:
    s_rf_str = f"sk_{CONFIG.get('s_kernel_sizes', (3,))[0]}_sdil_1"

# 提取 t 网络的感受野特征
t_rf_str = f"tk_{CONFIG.get('t_kernel_size', 3)}"

# 🌟 拼接终极版 base_name
base_name = (f"{CONFIG.get('type', 'asymmetric_prior_cnn')}_double_precision_"
             f"{CONFIG.get('double_precision', True)}_{CONFIG['L']}_coupling_layers_"
             f"{CONFIG['cnn_coupling_layers']}_depth_{CONFIG.get('branch_depth', 2)}_"
             f"s_ch_{CONFIG.get('s_channels', 128)}_t_ch_{CONFIG.get('t_channels', 32)}_"
             f"s_ly_{CONFIG.get('s_layers', 4)}_t_ly_{CONFIG.get('t_layers', 2)}_"
             f"{s_rf_str}_{t_rf_str}_"  # 👈 这里加入了 s 和 t 的感受野参数
             f"iterations_{CONFIG['iterations']}")

# 在最前面拼接你要求的前缀 (best_, latest_, phi_ensemble_, loss_)
save_path = f"best_{base_name}.pt"
loss_save_path = f"loss_{base_name}.npy"  # 统一格式，把 loss 移到了最前面
checkpoint_path = f"latest_{base_name}.pt"
phi_ensemble_save_path = f"phi_ensemble_{base_name}.npz"

CONFIG['save_path'] = save_path
CONFIG['loss_save_path'] = loss_save_path
CONFIG['checkpoint_path'] = checkpoint_path
# 🌟 [新增：物理可观测量] 专门保存每 100 步的接受率、phi 的期望值及误差
CONFIG['observables_save_path'] = f"observables_{base_name}.npz"
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
    def __init__(self, s_channels=128, s_layers=4, s_use_multi_kernel=True,
                 s_kernel_sizes=(3, 3), s_dilations=(1, 2),

                 # 🌟 t 网络的专属配置 (轻步兵 + 局域视野)
                 t_channels=32, t_layers=2,
                 t_kernel_size=3,
                 hidden_channels=8, num_hidden_layers=4, kernel_size=3, use_multi_kernel=False,
                 multi_kernel_sizes=(3, 5, 7), multi_kernel_dilations=(1, 1, 1), branch_depth=3):
        super().__init__()

        # --- 构造 s 网络的闭包函数 ---
        def build_s_net():
            layers = [nn.Conv2d(1, s_channels, kernel_size=3, padding=1, padding_mode='circular'), nn.LeakyReLU(0.01)]
            for _ in range(s_layers):
                if s_use_multi_kernel:
                    layers.append(MultiScaleResBlock(s_channels, kernel_sizes=s_kernel_sizes, dilations=s_dilations,
                                                     branch_depth=branch_depth))
                else:
                    layers.append(ResBlock(s_channels, kernel_size=s_kernel_sizes[0], branch_depth=branch_depth))
            layers.append(nn.Conv2d(s_channels, 1, kernel_size=1))
            return nn.Sequential(*layers)

        # --- 构造 t 网络的闭包函数 (强制只用单核，剥夺其宏观感受野) ---
        def build_t_net():
            # t 网络为了保证局域性，严格限制 padding 和 kernel
            t_pad = t_kernel_size // 2
            layers = [nn.Conv2d(1, t_channels, kernel_size=t_kernel_size, padding=t_pad, padding_mode='circular'),
                      nn.LeakyReLU(0.01)]
            for _ in range(t_layers):
                layers.append(ResBlock(t_channels, kernel_size=t_kernel_size, branch_depth=branch_depth))
            layers.append(nn.Conv2d(t_channels, 1, kernel_size=1))
            return nn.Sequential(*layers)

        # 实例化完全异构的两个子网络
        self.s_net = build_s_net()
        self.t_net = build_t_net()

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
# 4. 自由场先验
# ==========================================
class FreeFieldPrior(nn.Module):
    def __init__(self, L, m_sq_prior):
        super().__init__()
        self.L = L
        self.V = L * L
        p = torch.arange(L) * 2.0 * math.pi / L
        P1, P2 = torch.meshgrid(p, p, indexing='ij')
        K = m_sq_prior + 4.0 * torch.sin(P1 / 2.0) ** 2 + 4.0 * torch.sin(P2 / 2.0) ** 2
        self.register_buffer('sqrt_2K', torch.sqrt(2.0 * K).view(1, 1, L, L))
        self.register_buffer('log_det_factor', 0.5 * torch.sum(torch.log(2.0 * K)))
        self.register_buffer('const_factor', torch.tensor(0.5 * self.V * math.log(2.0 * math.pi)))

    def sample(self, batch_size):
        eta = torch.randn(batch_size, 1, self.L, self.L, device=self.sqrt_2K.device, dtype=self.sqrt_2K.dtype)
        eta_k = torch.fft.fftn(eta, dim=(-2, -1), norm="ortho")
        phi_k = eta_k / self.sqrt_2K
        phi_free = torch.fft.ifftn(phi_k, dim=(-2, -1), norm="ortho").real
        log_p_eta = -0.5 * torch.sum(eta ** 2, dim=(1, 2, 3)) - self.const_factor
        return phi_free, log_p_eta + self.log_det_factor


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
                # 🌟 传入非对称配置
                s_channels=config.get('s_channels', 128),
                t_channels=config.get('t_channels', 32),
                s_layers=config.get('s_layers', 4),
                t_layers=config.get('t_layers', 2),
                # 👇 🌟 关键修复：把原本旧的传参全部换成最新的 s_ 和 t_ 前缀参数
                s_use_multi_kernel=config.get('s_use_multi_kernel', True),
                s_kernel_sizes=config.get('s_kernel_sizes', (3, 3)),
                s_dilations=config.get('s_dilations', (1, 2)),
                t_kernel_size=config.get('t_kernel_size', 3),
                branch_depth=config.get('branch_depth', 3)
            ))

        # 🌟 修复：使用最新的非对称感受野参数
        if config.get('s_use_multi_kernel', True):
            s_arch = f"多尺度: {config.get('s_kernel_sizes', (3, 3))} | 空洞率: {config.get('s_dilations', (1, 2))}"
        else:
            s_arch = f"单核: {config.get('s_kernel_sizes', (3,))[0]}"
        t_arch = f"单核: {config.get('t_kernel_size', 3)}"

        print("=" * 70)
        print(f"🌟 非对称 Prior_CNN 物理流模型初始化完毕！")
        print(f"👉 总耦合层数: {self.total_layers} 层 (双步复用)")
        print(f"   │")
        print(f"   ├─ [容量分配] s 网络: {config.get('s_channels', 64)} 通道, {config.get('s_layers', 4)} 层")
        print(f"   │            t 网络: {config.get('t_channels', 32)} 通道, {config.get('t_layers', 2)} 层")
        print(f"   ├─ [感受野]   s 网络: {s_arch}")
        print(f"   │            t 网络: {t_arch}")
        print(f"   └─ 分支深度: {config.get('branch_depth', 2)}")
        print("=" * 70)

    def step_warmup(self, progress):
        """只负责放宽边界，防爆盾的开关由外部的 .train() 和 .eval() 决定"""
        """🌟 已废弃：采用永久 Leaky Clamp 后，不再需要动态放宽边界"""
        pass

    def forward(self, z, enforce_sym=False):
        phi = z
        log_det_jacobian = 0

        # 对齐 Z_2 脚本的参数复用逻辑：每个网络跑两次 (奇数格/偶数格)
        for net in self.context_nets:
            for step in range(2):
                current_mask = self.base_mask if step == 0 else (1.0 - self.base_mask)
                phi_frozen = current_mask * phi

                if enforce_sym:
                    # 🌟 核心逻辑：分别跑一次 z 和 -z
                    st_out_pos = net(phi_frozen)
                    st_out_neg = net(-phi_frozen)

                    s_out_pos, t_out_pos = st_out_pos[:, 0:1, :, :], st_out_pos[:, 1:2, :, :]
                    s_out_neg, t_out_neg = st_out_neg[:, 0:1, :, :], st_out_neg[:, 1:2, :, :]

                    # 强制 s 为偶函数: (s(z) + s(-z)) / 2
                    s_out = (s_out_pos + s_out_neg) / 2.0
                    # 强制 t 为奇函数: (t(z) - t(-z)) / 2
                    t_out = (t_out_pos - t_out_neg) / 2.0
                else:
                    # 原始逻辑
                    st_out = net(phi_frozen)
                    s_out, t_out = st_out[:, 0:1, :, :], st_out[:, 1:2, :, :]

                # 👇 🌟 治本之道：永久无条件施加泄漏截断 (Leaky Clamp)
                # 不管是训练还是评估，不管是第几步，永远兜住底线，同时保证梯度不断！
                # s_out = leaky_asymmetric_soft_clamp(s_out, self.s_bounds[0], self.s_bounds[1])
                # t_out = leaky_clamp(t_out, self.t_bounds[0], self.t_bounds[1])

                update_mask = 1.0 - current_mask

                # 🌟 统一仿射公式为逆向方程： y = (x - t) * exp(-s)
                phi = phi_frozen + update_mask * ((phi - t_out) * torch.exp(-s_out))
                log_det_jacobian += torch.sum(update_mask * (-s_out), dim=(1, 2, 3))

        return phi, log_det_jacobian


# ==========================================
# 🌟 治本神技：带有梯度泄漏的截断函数 (Leaky Clamp)
# ==========================================
def leaky_asymmetric_soft_clamp(x, min_val, max_val, leak=0.01):
    """用于 s 网络的非对称软截断，超出边界后保留 0.01 的微弱斜率"""
    pos_val = max_val * torch.tanh(x / max_val)
    neg_val = abs(min_val) * torch.tanh(x / abs(min_val))
    soft_clamped = torch.where(x >= 0, pos_val, neg_val)
    # 核心魔法：加上一层线性泄漏，保证梯度永远不为 0
    return soft_clamped + leak * (x - soft_clamped)

def leaky_clamp(x, min_val, max_val, leak=0.01):
    """用于 t 网络的硬截断，超出边界后保留 0.01 的微弱斜率"""
    hard_clamped = torch.clamp(x, min_val, max_val)
    return hard_clamped + leak * (x - hard_clamped)


# ==========================================
# ==========================================
# 6. 一体化评估核心 (修改版：增加 phi^1 到 phi^5 的在线期望计算)
# ==========================================
def run_mcmc_evaluation(model, prior, total_n=10000, batch_size=1024, enforce_sym=False):
    model.eval()
    total_n = (total_n // batch_size) * batch_size

    dtype = torch.float64 if CONFIG.get('double_precision', False) else torch.float32

    # 🌟 新增：预分配 Tensor 以存储所有的提案构型
    all_phis = torch.empty((total_n, 1, CONFIG['L'], CONFIG['L']), dtype=dtype, device=device)
    all_s = torch.empty(total_n, dtype=dtype, device=device)
    all_log_qs = torch.empty(total_n, dtype=dtype, device=device)

    dummy_progress = torch.tensor(1.0, device=device, dtype=dtype)

    # # 1. 正常采样一半的 z
    # z_half, _ = prior.sample(5000)
    #
    # # 2. 强行拼凑出完美对称的输入：[z, -z]
    # z_perfect = torch.cat([z_half, -z_half], dim=0)
    #
    # # 3. 让模型直接前向传播 (不开 MCMC)
    # model.eval()
    # with torch.no_grad():
    #     phi_perfect, _ = model(z_perfect, progress=torch.tensor(1.0), enforce_sym=True)
    #
    # # 4. 计算均值
    # print(f"完美对称输入下的 phi 均值: {torch.mean(phi_perfect).item():.8f}")

    with torch.no_grad():
        for i in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - i)
            z, log_p_z = prior.sample(current_batch)
            phi, log_det_J = model(z, dummy_progress, enforce_sym=enforce_sym)

            # 🌟 新增：记录下所有的提案 phi
            all_phis[i:i + current_batch] = phi
            all_s[i:i + current_batch] = compute_action(phi)
            all_log_qs[i:i + current_batch] = log_p_z - log_det_J

    accepted_count = 0
    curr_phi = all_phis[0]
    curr_s = all_s[0]
    curr_log_q = all_log_qs[0]

    log_rands = torch.log(torch.rand(total_n, device=device, dtype=dtype))

    # 🌟 新增：记录马尔可夫链中的每一个真实状态，用于后续计算观测值
    chain_phis = torch.empty_like(all_phis)
    chain_phis[0] = curr_phi

    for i in range(1, total_n):
        prop_phi = all_phis[i]
        prop_s = all_s[i]
        prop_log_q = all_log_qs[i]

        log_acc_ratio = (-prop_s - prop_log_q) - (-curr_s - curr_log_q)

        if log_rands[i] < log_acc_ratio:
            curr_phi = prop_phi
            curr_s = prop_s
            curr_log_q = prop_log_q
            accepted_count += 1

        # 记录当前 MCMC 步的构型
        chain_phis[i] = curr_phi

    # 🌟 新增：计算 phi^1 到 phi^5 的在线期望和朴素误差
    phi_powers_mean = []
    phi_powers_err = []

    for power in range(1, 6):
        # 先求空间维度的均值，得到时间序列 [total_n]
        pow_seq = (chain_phis ** power).mean(dim=(2, 3)).squeeze()
        # 求马尔可夫链的期望值
        mean_val = pow_seq.mean().item()
        # 朴素标准误 (这里是在线监控，不作复杂的 binning 去相关)
        err_val = (pow_seq.std() / math.sqrt(total_n)).item()

        phi_powers_mean.append(mean_val)
        phi_powers_err.append(err_val)

    model.train()
    # 🌟 修改：返回接受率的同时，返回期望值和误差列表
    return accepted_count / (total_n - 1), phi_powers_mean, phi_powers_err


# ==========================================
# 7. 训练主循环 (包含早停机制)
# ==========================================
def train():
    model = FlowModel(CONFIG).to(device)
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=abs(CONFIG['m_sq'])).to(device)

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
        # 👇 🌟 新增这段“无缝补偿”逻辑：
        # 如果当前已经跑过了退火阶段，直接把学习率锁定为你最新的底线
        # 👇 🌟 修复版断点恢复逻辑：
        if start_iteration > CONFIG['scheduler_steps']:
            # 情况 A：已经超过退火期，直接无缝锁定最低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = CONFIG['scheduler_min']
            print(
                f"🔧 已超过退火期 ({CONFIG['scheduler_steps']}步)，无缝接入最新最低学习率: {CONFIG['scheduler_min']:.2e}")
        else:
            # 👇 🌟 核心修复：情况 B：还在退火期内。必须强制把 optimizer 的起点记忆洗回 CONFIG['lr']
            # 否则会被断点里随机的历史学习率“劫持”，导致重新初始化的余弦曲线发生倒挂！
            for param_group in optimizer.param_groups:
                param_group['lr'] = CONFIG['lr']
                param_group['initial_lr'] = CONFIG['lr']
            print(f"🔄 处于退火期内，已重置 initial_lr 为 {CONFIG['lr']:.2e}，以确保余弦曲线正确生成。")
        history_loss = checkpoint.get('history_loss', [])
        ema_loss = checkpoint.get('ema_loss', None)
        best_loss = checkpoint.get('best_loss', min(history_loss) if history_loss else float('inf'))
        achieved_milestones = checkpoint.get('achieved_milestones', set())

    # 👇 🌟 修改点 2：添加读取 MCMC 历史数据的逻辑
    history_acc = []
    history_phi_means = []
    history_phi_errs = []
    mcmc_steps = []
    # 🌟 修改点：专门从可观测量文件中读取期望值和接受率
    if os.path.exists(CONFIG['observables_save_path']):
        try:
            npz_data = np.load(CONFIG['observables_save_path'])
            steps_array = npz_data['steps']
            # 截断失效的未来数据
            valid_idx = steps_array < start_iteration

            mcmc_steps = steps_array[valid_idx].tolist()
            history_acc = npz_data['acc'][valid_idx].tolist()
            history_phi_means = npz_data['phi_means'][valid_idx].tolist()
            history_phi_errs = npz_data['phi_errs'][valid_idx].tolist()
            print(f"✅ 成功加载外部物理观测记录，已对齐至第 {start_iteration - 1} 步。")
        except Exception as e:
            print(f"⚠️ 无法读取 {CONFIG['observables_save_path']}，将重新开始记录观测指标。错误: {e}")

    scheduler = None
    if CONFIG.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['scheduler_steps'],  # 🌟 修改：由 CONFIG['iterations'] 改为 CONFIG['scheduler_steps']
            eta_min=CONFIG['scheduler_min'],
            last_epoch=start_iteration - 1 if start_iteration > 1 else -1
        )

    if hasattr(torch, 'compile') and device.type == 'cuda':
        model.train()
        # compiled_model = torch.compile(model)
        compiled_model = model
    else:
        compiled_model = model

    model.train()
    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()
        z, log_p_z = prior.sample(CONFIG['batch_size'])

        # warmup_steps = CONFIG['warmup_steps']
        # progress_val = min(iteration / warmup_steps, 1.0)
        # model.step_warmup(progress_val)
        # progress_tensor = torch.tensor(progress_val, device=device, dtype=dtype)

        # 🌟 新增：动态判断当前是否需要开启强制对称性
        enforce_sym = CONFIG.get('enforce_z2_sym', False) and (iteration >= CONFIG.get('sym_start_iter', 0))

        # 🌟 修改：将 enforce_sym 传给模型
        phi, log_det_J = compiled_model(z, enforce_sym=enforce_sym)
        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi))
        loss.backward()

        # 🌟 修改点：方案 A - 全局宏观磁化率惩罚 (Global Magnetization Penalty)
        # 计算整个 Batch 内所有样本、所有格点的平均场值，并惩罚其平方
        # 🌟 修正：补偿体积因子，对齐 Action 的广延量级
        V = CONFIG['L'] * CONFIG['L']
        batch_mag = torch.mean(phi)
        loss_sym = V * (batch_mag ** 2)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("⚠️ 捕获到 NaN 梯度！跳过本次更新。")
            optimizer.zero_grad()
            continue
        else:
            optimizer.step()

        # 🌟 修改：只在前 25000 步推进 scheduler，之后停止推进并维持当前学习率
        if scheduler and iteration <= CONFIG['scheduler_steps']:
            scheduler.step()
        elif iteration == CONFIG['scheduler_steps'] + 1:
            print(
                f"🔄 调度器已完成前 {CONFIG['scheduler_steps']} 步降速，后续 10000 步学习率将固定在: {optimizer.param_groups[0]['lr']:.2e}")

        loss_val = loss.item()
        history_loss.append(loss_val)
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        if iteration % 100 == 0:
            # 🌟 新增：判断当前 Z_2 对称性是否处于开启状态
            z2_status = "ON" if enforce_sym else "OFF"

            print(
                f"迭代 {iteration:6d}/{CONFIG['iterations']} "
                f"| Z_2: {z2_status} "  # 🌟 加在这里，一目了然
                f"| 瞬时 Loss: {loss_val:.4f} "
                f"| 平滑 Loss: {ema_loss:.4f} "
                f"| Best: {best_loss:.4f} "
                f"| sym loss: {loss_sym:.4f} "  # 顺手加了 .4f 限制一下小数位数，版面更整洁
                f"| LR: {optimizer.param_groups[0]['lr']:.2e}")

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

        # ==========================================
        # 🌟 新增：每隔 2000 步保存一次带有当前 iteration 的检查点
        # ==========================================
        if iteration % 2000 == 0:
            # 提取公共的文件名核心部分，把 CONFIG['type'] 作为核心部分的开头
            base_name = (f"{CONFIG.get('type', 'asymmetric_prior_cnn')}_double_precision_"
                         f"{CONFIG.get('double_precision', True)}_{CONFIG['L']}_coupling_layers_"
                         f"{CONFIG['cnn_coupling_layers']}_depth_{CONFIG.get('branch_depth', 2)}_"
                         f"s_ch_{CONFIG.get('s_channels', 128)}_t_ch_{CONFIG.get('t_channels', 32)}_"
                         f"s_ly_{CONFIG.get('s_layers', 4)}_t_ly_{CONFIG.get('t_layers', 2)}_"
                         f"{s_rf_str}_{t_rf_str}_"  # 👈 这里加入了 s 和 t 的感受野参数
                         f"iterations_{CONFIG['iterations']}")
            # 动态拼接带有 iteration 数字的文件名
            iter_checkpoint_path = f"iter_{iteration}_{base_name}.pt"
            print(f"💾 [按步保存] 正在保存第 {iteration} 步的权重至: {iter_checkpoint_path}")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss_val, 'best_loss': best_loss,
                        'history_loss': history_loss,
                        'ema_loss': ema_loss,
                        'achieved_milestones': achieved_milestones},
                       iter_checkpoint_path)

        if iteration >= 10000 and iteration % 1000 == 0:
        # if iteration % 100 == 0:
            total_n = 50000
            if iteration % 5000 == 0:
                total_n = 100000
            print(f"\n{'=' * 50}")
            print(f"🚀 [迭代 {iteration}] 触发 MCMC 在线验证 (样本量: {total_n})...")

            # 🌟 修改：接收多出来的期望值和误差
            acc_rate, phi_means, phi_errs = run_mcmc_evaluation(
                compiled_model, prior, total_n=total_n, batch_size=CONFIG['batch_size'],
                enforce_sym=enforce_sym
            )

            print(f"📊 当前物理接受率: {acc_rate:.2%}")

            # 🌟 新增：按格式打印 1~5 次方的期望值
            print("------ Phi Powers Expectation ------")
            for p in range(1, 6):
                print(f"phi^{p}: {phi_means[p - 1]:.6f} ± {phi_errs[p - 1]:.6f}")

            print(f"{'=' * 50}\n")

            # 🌟 3. 将数据追加到列表中
            mcmc_steps.append(iteration)
            history_acc.append(acc_rate)
            history_phi_means.append(phi_means)
            history_phi_errs.append(phi_errs)
            # 🌟 独立保存物理观测期望值数据
            np.savez(CONFIG['observables_save_path'],
                     steps=np.array(mcmc_steps),
                     acc=np.array(history_acc),
                     phi_means=np.array(history_phi_means),
                     phi_errs=np.array(history_phi_errs))

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