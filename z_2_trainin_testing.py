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
        f"z_2_symmetry_shared_coupling_model_double_precision_*_"
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
    'batch_size': 512,
    'lr': 1e-3,
    'use_scheduler': True,
    'scheduler_min': 5e-6,
    'iterations': 35000,  # 提高总步数，靠 70% 接受率机制来提前早停
    'warmup_steps': 8000.0, # 退火步数，小于8000步会对参数进行数值限制
    'target_acc_ratio': 0.78,

    'cnn_coupling_layers': 6,
    'kernel_size': 3,
    'hidden_layers': 3,
    'branch_depth': 2,
    'hidden_channels': 32,
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

save_path = f"best_z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
loss_save_path = f"z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npy"
checkpoint_path = f"latest_z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
phi_ensemble_save_path = f"phi_ensemble_z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npz"

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


class LeakyTanh(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = 1.0 - alpha

    def forward(self, x):
        return self.alpha * x + self.beta * torch.tanh(x)


class CompiledWeightNormConv2d(nn.Module):
    """
    对 torch.compile 绝对友好的 WeightNorm 卷积。
    完全抛弃 PyTorch 官方的 parametrizations hook，用纯数学张量操作实现，
    彻底消除 AOTAutograd 在生成反向传播计算图时的 Graph Break。
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.dilation = dilation

        # 定义权重的方向 v (即普通的卷积参数，但不带 bias)
        self.weight_v = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))

        # 🌟 修复：使用底层的 平方 -> 求和 -> 开根号 替代 torch.norm
        with torch.no_grad():
            initial_norm = torch.sqrt(torch.sum(self.weight_v ** 2, dim=(1, 2, 3), keepdim=True))
        self.weight_g = nn.Parameter(initial_norm)

    def forward(self, x):
        # 🌟 修复：前向传播中同样使用最底层的数学算子计算 L2 范数
        v_norm = torch.sqrt(torch.sum(self.weight_v ** 2, dim=(1, 2, 3), keepdim=True))

        # 2. 计算最终使用的归一化权重: W = g * (v / ||v||)
        # 加 1e-8 防止除零引发 NaN
        w = self.weight_g * (self.weight_v / (v_norm + 1e-8))

        # 3. 使用底层 F.conv2d 执行卷积，编译器最喜欢这种静态的纯函数算子
        return F.conv2d(x, w, bias=None, stride=1, padding=0, dilation=self.dilation)


class ResBlock(nn.Module):
    # 移除了 num_groups 参数，因为不再使用 GroupNorm
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        padding = kernel_size // 2

        # 🌟 直接使用我们的自定义纯净版 WeightNorm
        self.conv1 = CompiledWeightNormConv2d(channels, channels, kernel_size=kernel_size)
        self.act1 = LeakyTanh()

        self.conv2 = CompiledWeightNormConv2d(channels, channels, kernel_size=kernel_size)
        self.act2 = LeakyTanh()

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


class MultiScaleResBlock(nn.Module):
    # 同样移除了 num_groups 参数
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), branch_depth=3):
        super().__init__()
        self.channels = channels  # 记录通道数用于方差缩放
        assert len(kernel_sizes) == len(dilations), "卷积核数量和空洞率数量必须严格匹配！"
        self.branches = nn.ModuleList()

        # 🌟 新增：记录每个分支对应的 padding 尺寸
        self.branch_pads = []

        for k, d in zip(kernel_sizes, dilations):
            assert k % 2 != 0, f"多尺度卷积核必须均为奇数！当前输入了偶数核: {k}"

            # 计算当前分支感受野所需的 padding
            pad_size = d * (k - 1) // 2
            self.branch_pads.append(pad_size)

            layers = nn.ModuleList()
            for _ in range(branch_depth):
                # 🌟 核心修改 1：彻底移除 padding_mode='circular'，强制设为 padding=0
                layers.append(CompiledWeightNormConv2d(channels, channels, kernel_size=k, dilation=d))
                # 2. 激活层 (移除了中间的 Norm)
                layers.append(LeakyTanh())
            self.branches.append(layers)

        # 融合卷积层：负责将多个多尺度特征压缩回原始通道数。
        self.fusion_conv = nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        # 🌟 流模型核心技巧：零初始化 (Zero-init for Flow Residuals)
        nn.init.zeros_(self.fusion_conv.weight)

    def forward(self, x):
        scale = math.sqrt(self.channels)
        outs = []

        # 🌟 核心修改 2：在遍历分支时，同时取出对应的 pad_size
        for branch, pad_size in zip(self.branches, self.branch_pads):
            out = x

            # F.pad 接收的参数格式是 (左, 右, 上, 下)
            pad_tuple = (pad_size, pad_size, pad_size, pad_size)

            for i in range(0, len(branch), 2):
                conv_layer = branch[i]
                act_layer = branch[i + 1]

                # 🌟 核心修改 3：在过卷积层之前，先用纯函数进行显式的 Circular Padding
                out = F.pad(out, pad=pad_tuple, mode='circular')

                # 正常过卷积层（此时底层不会再遇到 padding_mode 的黑盒）
                # (注意：保留了你原代码的写法，如果这里需要除以 scale 防爆，可以自行改回 / scale)
                out = act_layer(conv_layer(out))

            outs.append(out)

        # 沿通道维度拼接所有分支的结果
        fused = torch.cat(outs, dim=1)

        # 仅经过 1x1 卷积融合即可，不再经过 fusion_norm
        out = self.fusion_conv(fused)

        return x + out


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4, kernel_size=3, use_multi_kernel=False,
                 multi_kernel_sizes=(3, 5, 7), multi_kernel_dilations=(1, 1, 1), branch_depth=3):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1,
                                padding=1, bias=False))
        layers.append(LeakyTanh())

        for _ in range(num_hidden_layers):
            if use_multi_kernel:
                layers.append(MultiScaleResBlock(hidden_channels, kernel_sizes=multi_kernel_sizes,
                                                 dilations=multi_kernel_dilations, branch_depth=branch_depth))
            else:
                layers.append(ResBlock(hidden_channels, kernel_size=kernel_size))

        # 让主干网络直接输出 hidden_channels，不做通道压缩
        layers.append(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.net = nn.Sequential(*layers)

        # 🌟 根据切分比例，重新定义出口层的输入通道
        c = hidden_channels // 4

        # s_exit 接收 1/4 的通道 (h_even 的维度)
        self.s_exit_conv = nn.Conv2d(c, 1, kernel_size=1, stride=1, bias=True)

        # t_exit 接收剩下的 1/2 的独立通道
        self.t_exit_conv = nn.Conv2d(hidden_channels - 2 * c, 1, kernel_size=1, stride=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.net[0].weight, mean=0, std=0.1)
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.1)

        # 出口层严格清零，保证初始梯度满血复活！
        nn.init.zeros_(self.s_exit_conv.weight)
        nn.init.zeros_(self.s_exit_conv.bias)
        nn.init.zeros_(self.t_exit_conv.weight)

    def forward(self, x):
        out = self.net(x)

        # 🌟 破除耦合的核心：通道硬切分
        c = out.shape[1] // 4

        # 独立通道 1 和 2：专属用于构造偶函数 s
        h_s1 = out[:, 0:c, :, :]
        h_s2 = out[:, c:2 * c, :, :]

        # 独立通道 3：专属用于直接输出奇函数 t
        h_t_base = out[:, 2 * c:, :, :]

        # 分支 1：计算偶函数 s
        h_even = h_s1 * torch.tanh(h_s2)
        s_out = self.s_exit_conv(h_even)

        # 分支 2：计算奇函数 t (完全不依赖 h_s1 和 h_s2)
        t_out = self.t_exit_conv(h_t_base)

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
                        'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss, 'ema_loss': ema_loss},
                       checkpoint_path)

        if ema_loss < best_loss and iteration >= 10000:
            best_loss = ema_loss
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss, 'ema_loss': ema_loss},
                       save_path)

        # 🌟 新增：在线评估 MCMC 接受率与早停机制
        if iteration >= 10000 and iteration % 5000 == 0:
            print(f"\n{'=' * 50}")
            print(f"🚀 [迭代 {iteration}] 触发 MCMC 在线验证 (样本量: 10000)...")
            acc_rate = run_mcmc_evaluation(compiled_model, prior, total_n=10000, batch_size=CONFIG['batch_size'],)
            print(f"📊 当前物理接受率: {acc_rate:.2%}")
            print(f"{'=' * 50}\n")

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