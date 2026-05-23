import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
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
        f"bimodal_z_2_symmetry_shared_coupling_model_double_precision_*_"
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
# 1. 物理参数配置
# ==========================================
CONFIG = {
    'L': 14,
    'm_sq': -4.0,
    'lam': 5.113,
    'batch_size': 1024,
    'lr': 1e-3,
    'use_scheduler': True,
    'scheduler_min': 1e-5,
    'iterations': 25000,  # 提高总步数，靠 70% 接受率机制来提前早停
    'warmup_steps': 100.0, # 退火步数，小于8000步会对参数进行数值限制
    'target_acc_ratio': 0.78,

    'cnn_coupling_layers': 6,
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

save_path = f"best_bimodal_z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
loss_save_path = f"bimodal_z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npy"
checkpoint_path = f"latest_bimodal_z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
phi_ensemble_save_path = f"phi_ensemble_bimodal_z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npz"

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
# 2. 标量场理论的 Action 计算 (已深度优化)
# ==========================================
def compute_action(phi):
    phi_padded = F.pad(phi, pad=(1, 1, 1, 1), mode='circular')
    laplacian = F.conv2d(phi_padded, laplacian_kernel)
    action_density = phi * laplacian + CONFIG['m_sq'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 掩码生成与卷积上下文网络 (剔除了隐式 Padding 黑盒)
# ==========================================
def create_checkerboard_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()


class ResBlock(nn.Module):
    # 移除了 num_groups 参数，因为不再使用 GroupNorm
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.pad_size = kernel_size // 2

        # 🌟 直接使用我们的自定义纯净版 WeightNorm
        self.conv1 = CompiledWeightNormConv2d(channels, channels, kernel_size=kernel_size)
        self.act1 = OddELU()

        self.conv2 = CompiledWeightNormConv2d(channels, channels, kernel_size=kernel_size)
        self.act2 = OddELU()

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        # WeightNorm 会在第一次前向传播时根据数据自动调整权重尺度，
        # 因此不再强制需要 Kaiming 初始化，使用框架默认初始化即可。

        # 🌟 流模型核心技巧：残差分支零初始化 (Zero-init)
        # 确保网络在第 0 步时，这个 ResBlock 绝对等价于一个恒等映射 f(x) = x
        # 且由于没有 Norm 层拦截，现在梯度可以从深层畅通无阻地传回！
        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        scale = math.sqrt(self.channels)

        # 手动进行 Circular Padding
        pad = (self.pad_size, self.pad_size, self.pad_size, self.pad_size)

        out = F.pad(x, pad=pad, mode='circular')
        out = self.act1(self.conv1(out) / scale)

        out = F.pad(out, pad=pad, mode='circular')
        out = self.act2(self.conv2(out) / scale)

        out = F.pad(out, pad=pad, mode='circular')
        out = self.conv3(out)

        return x + out


class FullCapacityResBlock(nn.Module):
    """毫无保留的满血多尺度特征块：有 Bias，有标准非线性，完整跳跃连接"""
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), branch_depth=3):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList()
        self.branch_pads = []

        for k, d in zip(kernel_sizes, dilations):
            pad_size = d * (k - 1) // 2
            self.branch_pads.append(pad_size)
            layers = nn.ModuleList()
            for _ in range(branch_depth):
                # 🌟 满血回归：bias=True，标准的 LeakyReLU
                layers.append(nn.Conv2d(channels, channels, kernel_size=k, dilation=d, padding=0, bias=True))
                layers.append(nn.LeakyReLU(0.01))
            self.branches.append(layers)

        self.fusion_conv = nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1, bias=True)
        nn.init.normal_(self.fusion_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fusion_conv.bias)

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

        # ==========================================
        # s 和 t 现在是两个平起平坐的、完全无约束的满血网络！
        # ==========================================
        def build_full_capacity_net():
            layers = []
            # 入口层：带 Bias，补全了极其重要的 circular padding
            layers.append(
                nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=True))
            layers.append(nn.LeakyReLU(0.01))

            for _ in range(num_hidden_layers):
                if use_multi_kernel:
                    layers.append(
                        FullCapacityResBlock(hidden_channels, multi_kernel_sizes, multi_kernel_dilations, branch_depth))
                else:
                    raise NotImplementedError(
                        "当前未实现单核满血版的 ResBlock！请开启 use_multi_kernel=True 或自行实现。")
            # 出口层：同样带 Bias
            layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True))
            return nn.Sequential(*layers)

        self.s_net = build_full_capacity_net()
        self.t_net = build_full_capacity_net()

        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化：确保最终输出平稳起步
        nn.init.normal_(self.s_net[0].weight, mean=0, std=0.1)
        nn.init.zeros_(self.s_net[-1].weight)
        nn.init.zeros_(self.s_net[-1].bias)

        nn.init.normal_(self.t_net[0].weight, mean=0, std=0.1)
        nn.init.zeros_(self.t_net[-1].weight)
        nn.init.zeros_(self.t_net[-1].bias)

    def forward(self, x):
        # 🌟 极限性能优化：将 x 和 -x 拼在一个 Batch 里，榨干 GPU 算力，只做 1 次前向传播！
        x_both = torch.cat([x, -x], dim=0)

        s_both = self.s_net(x_both)
        t_both = self.t_net(x_both)

        # 拆分回正向结果和反向结果
        s_pos, s_neg = torch.chunk(s_both, 2, dim=0)
        t_pos, t_neg = torch.chunk(t_both, 2, dim=0)

        # 🌟 真正的物理数学之美：群平均对称化投影 (Reynolds Operator)
        s_out = (s_pos + s_neg) / 2.0  # 强制提取绝对偶函数分量
        t_out = (t_pos - t_neg) / 2.0  # 强制提取绝对奇函数分量

        return torch.cat([s_out, t_out], dim=1)


# ==========================================
# 4. 双峰自由场先验
# ==========================================
class BimodalFreeFieldPrior(nn.Module):
    def __init__(self, L, m_sq, lam):
        super().__init__()
        self.L = L
        self.V = L * L

        # 1. 物理学计算
        self.v = math.sqrt(-m_sq / (2.0 * lam))
        m_eff_sq = -2.0 * m_sq

        # 预先计算并缓存解析解所需的物理常数
        self.K_0 = m_eff_sq
        self.sqrt_2K_0 = math.sqrt(2.0 * self.K_0)

        # 2. 构造动量空间算子
        p = torch.arange(L) * 2.0 * math.pi / L
        P1, P2 = torch.meshgrid(p, p, indexing='ij')
        K = m_eff_sq + 4.0 * torch.sin(P1 / 2.0) ** 2 + 4.0 * torch.sin(P2 / 2.0) ** 2

        self.register_buffer('v_tensor', torch.tensor(self.v))
        self.register_buffer('sqrt_2K', torch.sqrt(2.0 * K).view(1, 1, L, L))
        self.register_buffer('log_det_factor', 0.5 * torch.sum(torch.log(2.0 * K)))
        self.register_buffer('const_factor', torch.tensor(0.5 * self.V * math.log(2.0 * math.pi)))

    def sample(self, batch_size):
        device = self.sqrt_2K.device
        dtype = self.sqrt_2K.dtype

        # 1. 采样标准的自由场波动 (天然无误差)
        eta = torch.randn(batch_size, 1, self.L, self.L, device=device, dtype=dtype)
        eta_k = torch.fft.fftn(eta, dim=(-2, -1), norm="ortho")
        phi_k = eta_k / self.sqrt_2K
        phi_free = torch.fft.ifftn(phi_k, dim=(-2, -1), norm="ortho").real

        # 2. 全局平移生成物理构型
        signs = torch.randint(0, 2, size=(batch_size, 1, 1, 1), device=device).to(dtype) * 2.0 - 1.0
        phi_bimodal = phi_free + signs * self.v_tensor

        # 3. 极速、零误差的 GMM Log 概率解析计算
        # (a) 样本在其“出生”势阱中的精确概率
        log_p_s = -0.5 * torch.sum(eta ** 2, dim=(1, 2, 3)) - self.const_factor + self.log_det_factor

        # (b) O(1) 算出样本在“另一个”势阱中的概率（彻底消灭 FFT 误差）
        eta_sum = torch.sum(eta, dim=(1, 2, 3))
        s_val = signs.view(batch_size)

        delta_p = -2.0 * s_val * self.v * self.sqrt_2K_0 * eta_sum - 4.0 * self.V * (self.v ** 2) * self.K_0
        log_p_other = log_p_s + delta_p

        # (c) 混合高斯分布的最终概率
        log_p_z = torch.logaddexp(log_p_s, log_p_other) - math.log(2.0)

        return phi_bimodal, log_p_z


# ==========================================
# 5. 流模型
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config['L']
        self.cnn_layers = config['cnn_coupling_layers']
        self.total_layers = self.cnn_layers
        self.register_buffer('base_mask', create_checkerboard_mask(self.L))
        self.context_nets = nn.ModuleList()
        # 提取当前设备类型和精度
        dtype = torch.float64 if config.get('double_precision', False) else torch.float32

        # 注册时使用动态 dtype
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
        print(f"🌟 物理流模型初始化完毕！")
        print(f"👉 总耦合层数: {self.total_layers} 层 (特征通道数: {config['hidden_channels']})")
        print(f"   │")
        print(f"   ├─ [UV 物理] 纯 CNN 局域重整化: 前 {self.cnn_layers} 层")
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

        for net in self.context_nets:
            for step in range(2):
                current_mask = self.base_mask if step == 0 else (1.0 - self.base_mask)
                phi_frozen = current_mask * phi

                st_out = net(phi_frozen)
                s_out, t_out = st_out[:, 0:1, :, :], st_out[:, 1:2, :, :]

                # 🌟 核心改造：完全消除动态 Python if 分支
                # self.training 不会引发问题，因为 PyTorch 自动为 train 和 eval 编译两张独立的静态图
                if self.training and progress is not None:
                    # 1. 此时 progress 是一个 Tensor。生成一个标量布尔掩码，并扩展维度以支持广播
                    is_warmup = (progress < 1.0).view(1, 1, 1, 1)

                    # 2. 正常计算裁剪结果
                    s_clamped = asymmetric_soft_clamp(s_out, self.s_bounds[0], self.s_bounds[1])
                    t_clamped = torch.clamp(t_out, self.t_bounds[0], self.t_bounds[1])

                    # 3. 使用 torch.where 让底层 Kernel 自动选择（彻底消灭 Graph Break）
                    s_out = torch.where(is_warmup, s_clamped, s_out)
                    t_out = torch.where(is_warmup, t_clamped, t_out)

                update_mask = 1.0 - current_mask
                phi = phi_frozen + update_mask * ((phi - t_out) * torch.exp(-s_out))
                log_det_jacobian += torch.sum(update_mask * (-s_out), dim=(1, 2, 3))

        return phi, log_det_jacobian


def asymmetric_soft_clamp(x, min_val, max_val):
    pos_val = max_val * torch.tanh(x / max_val)
    neg_val = abs(min_val) * torch.tanh(x / abs(min_val))
    return torch.where(x >= 0, pos_val, neg_val)


def clamp_t(x, min_val, max_val):
    return torch.clamp(x, min_val, max_val)


# ==========================================
# 6. 一体化评估核心 (支持轻量验证 & 深度生产)
# ==========================================
def run_mcmc_evaluation(model, prior, total_n=10000, batch_size=1024):
    """
    极简版在线验证 (纯 GPU 流水线)
    专注验证接受率，无显存堆积，完全规避 CPU/GPU 内存通信。
    """
    model.eval()

    # 强制修正，消灭尾部不规则 Batch
    total_n = (total_n // batch_size) * batch_size

    # 1. 预先在 GPU 上分配 1D 标量空间 (1万个浮点数仅占用不到 1MB)
    dtype = torch.float64 if CONFIG.get('double_precision', False) else torch.float32
    all_s = torch.empty(total_n, dtype=dtype, device=device)
    all_log_qs = torch.empty(total_n, dtype=dtype, device=device)

    # 生成一个占位张量，保持与训练期签名一致
    dummy_progress = torch.tensor(1.0, device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - i)

            z, log_p_z = prior.sample(current_batch)
            phi, log_det_J = model(z, dummy_progress)

            # 按批次算完 Action 直接存入 GPU 的 1D 数组
            # 此时局部的 4D 张量 phi 随循环结束被自动回收，永不 OOM
            all_s[i:i + current_batch] = compute_action(phi)
            all_log_qs[i:i + current_batch] = log_p_z - log_det_J

    # ==========================================
    # 2. 严格的马尔可夫链演化推演 (满足细致平衡)
    # ==========================================
    accepted_count = 0

    # 确立链的初始状态（由第 0 个提案初始化）
    curr_s = all_s[0]
    curr_log_q = all_log_qs[0]

    # 🔧 修复点：一次性在 GPU 上分配好所有时间步所需的独立同分布随机数 (O(1) 开销)
    log_rands = torch.log(torch.rand(total_n, device=device, dtype=dtype))

    for i in range(1, total_n):
        prop_s = all_s[i]
        prop_log_q = all_log_qs[i]

        # Delta = S_eff(curr) - S_eff(prop)
        log_acc_ratio = (-prop_s - prop_log_q) - (-curr_s - curr_log_q)

        # 🔧 修复点：按时间步 i 精准消耗对应的预生成随机数
        if log_rands[i] < log_acc_ratio:
            curr_s = prop_s
            curr_log_q = prop_log_q
            accepted_count += 1

    # 将模型无缝切回训练模式
    model.train()

    # 接受率 = 链成功跳转的次数 / 总跳转尝试次数 (total_n - 1)
    return accepted_count / (total_n - 1)


# ==========================================
# 7. 训练主循环 (包含早停机制)
# ==========================================
def train():
    model = FlowModel(CONFIG).to(device)
    prior = BimodalFreeFieldPrior(L=CONFIG['L'], m_sq=CONFIG['m_sq'],lam=CONFIG['lam']).to(device)

    if CONFIG['double_precision']:
        model = model.double()
        prior = prior.double()

    eps_val = 1e-15 if CONFIG.get('double_precision', False) else 1e-8
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], eps=eps_val)

    history_loss = []
    best_loss = float('inf')
    ema_loss = None
    start_iteration = 1

    # 🌟 新增：定义我们需要追踪的接受率里程碑 (降序排列方便逻辑判断)
    target_milestones = sorted([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75], reverse=True)
    achieved_milestones = set()  # 用于追踪已经达成的里程碑

    old_checkpoint_path = auto_find_latest_checkpoint(CONFIG)
    if old_checkpoint_path and os.path.exists(old_checkpoint_path):
        print(f"从 {old_checkpoint_path} 恢复训练...")
        checkpoint = torch.load(old_checkpoint_path, map_location=device)

        # 剥离编译前缀以便完美加载
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

        # 🌟 新增：从 checkpoint 恢复已达成的里程碑
        achieved_milestones = checkpoint.get('achieved_milestones', set())

    # 🌟 严苛修复：在时间轴位置明确后，严格对齐相位并重构调度器
    scheduler = None
    if CONFIG.get('use_scheduler', True):
        # 传入 start_iteration - 1 作为 last_epoch，
        # PyTorch 会自动根据原有的 initial_lr 和新的 T_max，计算出当前完美平滑的真实学习率
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['iterations'],  # 直接使用延长后的新目标步数
            eta_min=CONFIG['scheduler_min'],
            last_epoch=start_iteration - 1 if start_iteration > 1 else -1
        )

    if hasattr(torch, 'compile') and device.type == 'cuda':
        # 强制指定 fullgraph=False 允许标准的常数分支拍平，
        # 并提前切换状态让编译器做好双图缓存准备
        model.train()
        compiled_model = torch.compile(model)
    else:
        compiled_model = model



    model.train()
    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()
        z, log_p_z = prior.sample(CONFIG['batch_size'])

        # 🌟 动态边界退火：前 15000 步平滑放开限制，之后完全解除
        warmup_steps = CONFIG['warmup_steps']
        progress_val = min(iteration / warmup_steps, 1.0)

        # 这个操作在图外进行，不影响编译
        model.step_warmup(progress_val)

        # 🌟 将 progress 打包成 Tensor 传入编译后的模型
        progress_tensor = torch.tensor(progress_val, device=device, dtype=dtype)

        # 使用编译后的模型
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
                        'achieved_milestones': achieved_milestones},  # 🌟 新增
                       checkpoint_path)

        if ema_loss < best_loss and iteration >= 10000:
            best_loss = ema_loss
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss, 'ema_loss': ema_loss,
                        'achieved_milestones': achieved_milestones},  # 🌟 新增
                       save_path)

        # 🌟 新增：在线评估 MCMC 接受率与早停机制
        if iteration >= 10000 and iteration % 2000 == 0:
            print(f"\n{'=' * 50}")
            print(f"🚀 [迭代 {iteration}] 触发 MCMC 在线验证 (样本量: 10000)...")
            acc_rate = run_mcmc_evaluation(compiled_model, prior, total_n=10000, batch_size=CONFIG['batch_size'],)
            print(f"📊 当前物理接受率: {acc_rate:.2%}")
            print(f"{'=' * 50}\n")

            # 🌟 新增：检查是否触发了特定接受率里程碑
            for m in target_milestones:
                if acc_rate >= m and m not in achieved_milestones:
                    print(f"⭐ 达成里程碑！接受率首次突破 {m:.0%} (当前 {acc_rate:.2%})，正在保存专属模型...")

                    # 构造包含接受率的文件名，剥离 "latest_" 前缀并加入 acc_XXpercent_
                    base_name = checkpoint_path.replace('latest_', '')
                    milestone_path = f"acc_{int(m * 100)}percent_{base_name}"

                    # 记录该里程碑以及所有更低的里程碑为已达成 (防止跃级导致重复触发)
                    for lm in target_milestones:
                        if lm <= m:
                            achieved_milestones.add(lm)

                    # 保存里程碑模型
                    torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                                'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss,
                                'ema_loss': ema_loss,
                                'achieved_milestones': achieved_milestones},
                               milestone_path)
                    break  # 仅触发当前达到的最高里程碑，跳出循环

            if acc_rate >= CONFIG['target_acc_ratio']:
                torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'loss': loss_val,
                            'best_loss': best_loss,
                            'history_loss': history_loss,
                            'ema_loss': ema_loss, }, 'mcmc_success_'+checkpoint_path)
                print(f"🎉 成功达标！接受率已达到 {acc_rate:.2%} (>= {CONFIG['target_acc_ratio']:.0%})，提前结束训练阶段！")
                break

    print("✅ 训练流水线结束！")
    return model


if __name__ == "__main__":
    if CONFIG.get('double_precision', False): torch.set_default_dtype(torch.float64)

    # 1. 执行训练并返回最新模型
    final_model = train()