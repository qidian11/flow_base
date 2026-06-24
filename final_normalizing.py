import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import os
import glob

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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


def fmt_cfg(val):
    """将列表参数压缩为紧凑的字符串，例如 [1, 1, 2, 2, 1, 1] -> '1x2-2x2-1x2'"""
    if not isinstance(val, (list, tuple)):
        return str(val)

    compressed = []
    count = 1
    for i in range(1, len(val)):
        if val[i] == val[i - 1]:
            count += 1
        else:
            compressed.append(f"{val[i - 1]}x{count}" if count > 1 else str(val[i - 1]))
            count = 1
    compressed.append(f"{val[-1]}x{count}" if count > 1 else str(val[-1]))

    return "-".join(compressed)


def auto_find_latest_checkpoint(config):
    tk_str = '_'.join(map(str, config.get('trunk_kernel_sizes', (3, 3))))
    tdil_str = '_'.join(map(str, config.get('trunk_dilations', (1, 2))))
    trunk_rf = f"trk_{tk_str}_dil_{tdil_str}"

    # 🌟 使用 fmt_cfg 包装列表参数，确保正则匹配的文件名是压缩后的格式
    base_pattern = (
        f"{config.get('type', 'shared_trunk_prior_cnn')}_dp_*_"
        f"L{config['L']}_c{config['cnn_coupling_layers']}_d*_"
        f"TrCh{fmt_cfg(config.get('trunk_channels'))}_Ly{fmt_cfg(config.get('trunk_layers'))}_{trunk_rf}_"
        f"Sh{fmt_cfg(config.get('s_head_channels'))}L{fmt_cfg(config.get('s_head_layers'))}_"
        f"Th{fmt_cfg(config.get('t_head_channels'))}L{fmt_cfg(config.get('t_head_layers'))}_"
        f"iter_*.pt"
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
# 1. 物理参数配置
# ==========================================
# ==========================================
# 1. 物理参数配置
# ==========================================
CONFIG = {
    # 'type': 'shared_trunk_prior_cnn8_m_free',
    'type': 'final_normalizing',
    'L': 14,
    'm_sq': -4.0,
    'lam': 5.113,
    'batch_size': 1024,
    'lr': 1e-3,
    'use_scheduler': True,
    'scheduler_min': 1e-6,
    'iterations': 100000,
    'scheduler_steps': 60000,
    'warmup_steps': 12000.0,
    'target_acc_ratio': 0.78,

    'enforce_z2_sym': False,
    'sym_start_iter': 30000,

    'cnn_coupling_layers': 12,
    'branch_depth': 2,
    'double_precision': False,

    # 🌟 核心修改：支持列表，按 U-Net "沙漏" 风格设计，中间层更深更宽
    # 如果用单个整数（如 96），则兼容旧版，所有层全部为 96
    # 'trunk_channels': [128, 128, 128, 128, 128, 128],
    'trunk_channels': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], # 64通道参数
    # 'trunk_channels': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'trunk_layers': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'trunk_use_multi_kernel': True,
    'trunk_kernel_sizes': (3, 3),# 64通道参数
    'trunk_dilations': (1, 2),# 64通道参数
    # 'trunk_kernel_sizes': (3, ),
    # 'trunk_dilations': (1, ),

    # 🌟 S 分支也支持逐层调控，首尾较浅，中间较深
    's_head_channels': [96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96],
    # 's_head_channels': [8, 8, 8, 8, 8, 8],
    's_head_layers': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    's_head_kernel_size': 3,

    # 🌟 T 分支同理
    't_head_channels': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    # 't_head_channels': [8, 8, 8, 8, 8, 8],
    't_head_layers': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    't_kernel_size': 3,
}

# ==========================================
# 动态生成文件名核心
# ==========================================
# 提取感受野特征
tk_str = '_'.join(map(str, CONFIG.get('trunk_kernel_sizes', (3, 3))))
tdil_str = '_'.join(map(str, CONFIG.get('trunk_dilations', (1, 2))))
trunk_rf = f"trk_{tk_str}_dil_{tdil_str}"

# 🌟 使用 fmt_cfg 包装列表参数
base_name = (f"{CONFIG.get('type', 'shared_trunk_prior_cnn')}_dp_{CONFIG.get('double_precision', True)}_"
             f"L{CONFIG['L']}_c{CONFIG['cnn_coupling_layers']}_d{CONFIG.get('branch_depth', 2)}_"
             f"TrCh{fmt_cfg(CONFIG.get('trunk_channels'))}_Ly{fmt_cfg(CONFIG.get('trunk_layers'))}_{trunk_rf}_"
             f"Sh{fmt_cfg(CONFIG.get('s_head_channels'))}L{fmt_cfg(CONFIG.get('s_head_layers'))}_"
             f"Th{fmt_cfg(CONFIG.get('t_head_channels'))}L{fmt_cfg(CONFIG.get('t_head_layers'))}_"
             f"iter_{CONFIG['iterations']}")

CONFIG['base_name'] = base_name

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
    # 🌟 关键修改：动态克隆一个与输入张量完全一致的卷积核
    # 这样无论是训练(FP32)还是验证(FP64)，都不会报错
    adaptive_kernel = laplacian_kernel.to(dtype=phi.dtype, device=phi.device)

    laplacian = F.conv2d(phi_padded, adaptive_kernel)
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
        # 👇 🌟 必须加上这两行！驯服大步长的核心！
        nn.init.zeros_(self.block[-2].weight)
        if self.block[-2].bias is not None:
            nn.init.zeros_(self.block[-2].bias)

    def forward(self, x):
        # 🌟 现在的残差连接极其简洁
        return x + self.block(x)


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), branch_depth=3):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList()

        for k, d in zip(kernel_sizes, dilations):
            # 精准计算 padding 大小，配合 circular mode 完美保持平移不变性
            pad_size = d * (k - 1) // 2
            layers = nn.ModuleList()
            for _ in range(branch_depth):
                # 🚀 优化：利用 PyTorch 底层的 padding_mode='circular'，彻底抛弃 F.pad
                layers.append(nn.Conv2d(channels, channels, kernel_size=k, dilation=d,
                                        padding=pad_size, padding_mode='circular', bias=False))
                layers.append(nn.LeakyReLU(0.01))
            self.branches.append(layers)

        self.fusion_conv = nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fusion_conv.weight)

    def forward(self, x):
        outs = []
        for branch in self.branches:
            out = x
            for i in range(0, len(branch), 2):
                conv_layer = branch[i]
                act_layer = branch[i + 1]
                # 🚀 优化：前向传播无比干净，没有任何额外的 Tensor 切片和拼接
                out = act_layer(conv_layer(out))
            outs.append(out)

        fused = torch.cat(outs, dim=1)
        out = self.fusion_conv(fused)
        return x + out


class ConvContextNet(nn.Module):
    def __init__(self,
                 trunk_channels=64, trunk_layers=3,
                 trunk_use_multi_kernel=True, trunk_kernel_sizes=(3, 3), trunk_dilations=(1, 2),
                 s_head_channels=32, s_head_layers=1, s_head_kernel_size=3,  # 👈 接收 s_kernel
                 t_head_channels=32, t_head_layers=1, t_kernel_size=3,
                 branch_depth=2):
        super().__init__()

        self.s_head_layers = s_head_layers
        self.t_head_layers = t_head_layers

        # --- 🌟 1. 构造共享主干网络 (Shared Trunk) ---
        trunk_list = [nn.Conv2d(1, trunk_channels, kernel_size=3, padding=1, padding_mode='circular'),
                      nn.LeakyReLU(0.01)]
        for _ in range(trunk_layers):
            if trunk_use_multi_kernel:
                trunk_list.append(
                    MultiScaleResBlock(trunk_channels, kernel_sizes=trunk_kernel_sizes, dilations=trunk_dilations,
                                       branch_depth=branch_depth))
            else:
                trunk_list.append(
                    ResBlock(trunk_channels, kernel_size=trunk_kernel_sizes[0], branch_depth=branch_depth))
        self.trunk_net = nn.Sequential(*trunk_list)

        # --- 🌟 2. 构造 S 的独立分支 (支持 0 层直通) ---
        s_pad = s_head_kernel_size // 2
        if s_head_layers == 0:
            # 如果为0，直接用一层卷积把 trunk 特征映射为1通道输出 (s)
            self.s_head = nn.Conv2d(trunk_channels, 1, kernel_size=s_head_kernel_size, padding=s_pad,
                                    padding_mode='circular')
        else:
            s_head_list = [nn.Conv2d(trunk_channels, s_head_channels, kernel_size=s_head_kernel_size, padding=s_pad,
                                     padding_mode='circular'), nn.LeakyReLU(0.01)]
            for _ in range(s_head_layers):
                s_head_list.append(ResBlock(s_head_channels, kernel_size=s_head_kernel_size, branch_depth=branch_depth))
            s_head_list.append(nn.Conv2d(s_head_channels, 1, kernel_size=1))
            self.s_head = nn.Sequential(*s_head_list)

        # --- 🌟 3. 构造 T 的独立分支 (支持 0 层直通) ---
        t_pad = t_kernel_size // 2
        if t_head_layers == 0:
            # 如果为0，直接用一层卷积把 trunk 特征映射为1通道输出 (t)
            self.t_head = nn.Conv2d(trunk_channels, 1, kernel_size=t_kernel_size, padding=t_pad,
                                    padding_mode='circular')
        else:
            t_head_list = [nn.Conv2d(trunk_channels, t_head_channels, kernel_size=t_kernel_size, padding=t_pad,
                                     padding_mode='circular'), nn.LeakyReLU(0.01)]
            for _ in range(t_head_layers):
                t_head_list.append(ResBlock(t_head_channels, kernel_size=t_kernel_size, branch_depth=branch_depth))
            t_head_list.append(nn.Conv2d(t_head_channels, 1, kernel_size=1))
            self.t_head = nn.Sequential(*t_head_list)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.trunk_net[0].weight, mean=0, std=0.1)

        # 智能适配 0 层和多层的 S 网络初始化
        if self.s_head_layers == 0:
            nn.init.zeros_(self.s_head.weight)
            nn.init.zeros_(self.s_head.bias)
        else:
            nn.init.normal_(self.s_head[0].weight, mean=0, std=0.05)
            nn.init.zeros_(self.s_head[-1].weight)
            nn.init.zeros_(self.s_head[-1].bias)

        # 智能适配 0 层和多层的 T 网络初始化
        if self.t_head_layers == 0:
            nn.init.zeros_(self.t_head.weight)
            nn.init.zeros_(self.t_head.bias)
        else:
            nn.init.normal_(self.t_head[0].weight, mean=0, std=0.05)
            nn.init.zeros_(self.t_head[-1].weight)
            nn.init.zeros_(self.t_head[-1].bias)

    def forward(self, x):
        shared_feat = self.trunk_net(x)
        s_out = self.s_head(shared_feat)
        t_out = self.t_head(shared_feat)
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
        self.register_buffer('s_bounds', torch.tensor([-0.2, 3.0], dtype=dtype))
        self.register_buffer('t_bounds', torch.tensor([-3.0, 3.0], dtype=dtype))

        # 🌟 新增：提取单层配置的辅助函数
        def get_layer_cfg(key, default, layer_idx):
            val = config.get(key, default)
            if isinstance(val, (list, tuple)):
                assert len(
                    val) == self.cnn_layers, f"❌ 参数 {key} 的列表长度({len(val)})必须等于 cnn_coupling_layers({self.cnn_layers})"
                return val[layer_idx]
            return val  # 兼容单数字配置

        for i in range(self.cnn_layers):
            self.context_nets.append(ConvContextNet(
                trunk_channels=get_layer_cfg('trunk_channels', 64, i),
                trunk_layers=get_layer_cfg('trunk_layers', 3, i),
                trunk_use_multi_kernel=config.get('trunk_use_multi_kernel', True),
                trunk_kernel_sizes=config.get('trunk_kernel_sizes', (3, 3)),
                trunk_dilations=config.get('trunk_dilations', (1, 2)),

                s_head_channels=get_layer_cfg('s_head_channels', 32, i),
                s_head_layers=get_layer_cfg('s_head_layers', 1, i),

                t_head_channels=get_layer_cfg('t_head_channels', 32, i),
                t_head_layers=get_layer_cfg('t_head_layers', 1, i),

                t_kernel_size=config.get('t_kernel_size', 3),
                s_head_kernel_size=config.get('s_head_kernel_size', 3),
                branch_depth=config.get('branch_depth', 2)
            ))

        # 🌟 修改控制台打印，展示动态结构
        print("=" * 70)
        print(f"🌟 Variable Shared-Trunk (Y-Net) 物理流模型初始化完毕！")
        print(f"👉 耦合层数: {self.total_layers} 层 (双步复用) | 分支深度: {config.get('branch_depth', 2)}")
        print("   ├─ [层级分布日志]:")
        for i in range(self.cnn_layers):
            tr_ch = get_layer_cfg('trunk_channels', 64, i)
            tr_ly = get_layer_cfg('trunk_layers', 3, i)
            s_ch = get_layer_cfg('s_head_channels', 32, i)
            s_ly = get_layer_cfg('s_head_layers', 1, i)
            t_ch = get_layer_cfg('t_head_channels', 32, i)
            t_ly = get_layer_cfg('t_head_layers', 1, i)
            print(
                f"   │  └─ 第 {i + 1} 层: Trunk({tr_ch}ch, {tr_ly}L) | S_Head({s_ch}ch, {s_ly}L) | T_Head({t_ch}ch, {t_ly}L)")
        print("=" * 70)

    def step_warmup(self, progress):
        """永远不撤除防爆盾，最高放宽到物理安全极限"""
        clamped_progress = min(progress, 1.0)
        self.s_bounds[0].fill_(-0.2 - 0.3 * clamped_progress)
        self.s_bounds[1].fill_(3.0 + 2.0 * clamped_progress)
        self.t_bounds[0].fill_(-3.0 - 5.0 * clamped_progress)
        self.t_bounds[1].fill_(3.0 + 5.0 * clamped_progress)

        """只负责放宽边界，防爆盾的开关由外部的 .train() 和 .eval() 决定"""
        # if progress < 1.0:
        #     self.s_bounds[0].fill_(-0.2 - 0.3 * progress)
        #     self.s_bounds[1].fill_(3.0 + 2.0 * progress)
        #     self.t_bounds[0].fill_(-3.0 - 5.0 * progress)
        #     self.t_bounds[1].fill_(3.0 + 5.0 * progress)
        # else:
        #     self.s_bounds[0].fill_(-100.0)
        #     self.s_bounds[1].fill_(100.0)
        #     self.t_bounds[0].fill_(-100.0)
        #     self.t_bounds[1].fill_(100.0)

    def forward(self, z, progress=None, enforce_sym=False):
        phi = z
        log_det_jacobian = 0

        # 🌟 核心修改：去除双循环，直接遍历 context_nets，让 6 个网络独立跑 6 步
        for step, net in enumerate(self.context_nets):
            # 偶数步使用基础掩码，奇数步使用反转掩码
            current_mask = self.base_mask if step % 2 == 0 else (1.0 - self.base_mask)
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

            # 🌟 无条件、永久生效的绝对防御！
            s_clamped = asymmetric_soft_clamp(s_out, self.s_bounds[0], self.s_bounds[1])
            t_clamped = self.t_bounds[1] * torch.tanh(t_out / self.t_bounds[1])
            s_out = s_clamped
            t_out = t_clamped

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

    # ================= ⬇️ 将这段粘贴到原来删除的位置 ⬇️ =================
    # 🌟 修改：MCMC 提速黑魔法 - NumPy 批量切片法
    # 1. 把所有一维的标量数据拉回 CPU 内存 (10万个 float 瞬间完成)
    s_np = all_s.cpu().numpy()
    log_q_np = all_log_qs.cpu().numpy()

    # 注意这里直接在 CPU 上生成随机数，省去了 GPU 通信
    log_rands_np = np.log(np.random.rand(total_n))

    accepted_indices = np.zeros(total_n, dtype=int)
    accepted_count = 0

    curr_s_val = s_np[0]
    curr_log_q_val = log_q_np[0]
    accepted_indices[0] = 0

    # 2. 在纯 CPU 内存里跑循环，只算标量加减，极其快速
    for i in range(1, total_n):
        prop_s_val = s_np[i]
        prop_log_q_val = log_q_np[i]

        log_acc_ratio = (-prop_s_val - prop_log_q_val) - (-curr_s_val - curr_log_q_val)

        if log_rands_np[i] < log_acc_ratio:
            curr_s_val = prop_s_val
            curr_log_q_val = prop_log_q_val
            accepted_indices[i] = i
            accepted_count += 1
        else:
            accepted_indices[i] = accepted_indices[i - 1]  # 拒绝则保留上一步的索引

    # 3. 回到 GPU，一次性切片提取完整的构型链 (消灭了所有对构型的 for 循环)
    idx_tensor = torch.tensor(accepted_indices, device=device, dtype=torch.long)
    chain_phis = all_phis[idx_tensor]

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
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=0.6005269985).to(device)
    # prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=abs(CONFIG['m_sq'])).to(device)

    if CONFIG['double_precision']:
        model = model.double()
        prior = prior.double()

    eps_val = 1e-15 if CONFIG.get('double_precision', False) else 1e-8
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], eps=eps_val)

    history_loss = []
    best_loss = float('inf')
    ema_loss = None
    start_iteration = 1

    target_milestones = sorted([0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75], reverse=True)
    achieved_milestones = set()

    old_checkpoint_path = auto_find_latest_checkpoint(CONFIG)
    if old_checkpoint_path and os.path.exists(old_checkpoint_path):
        print(f"从 {old_checkpoint_path} 恢复训练...")
        checkpoint = torch.load(old_checkpoint_path, map_location=device)

        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(clean_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 👇 🌟 【终极清洗法】：强制重写底层的初始值，杜绝任何旧档的污染
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = CONFIG['lr']
            param_group['lr'] = CONFIG['lr']

        if CONFIG.get('double_precision', False):
            for group in optimizer.param_groups:
                group['eps'] = 1e-15
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point(): state[k] = v.double()

        start_iteration = checkpoint['iteration'] + 1
        history_loss = checkpoint.get('history_loss', [])
        ema_loss = checkpoint.get('ema_loss', None)
        best_loss = checkpoint.get('best_loss', min(history_loss) if history_loss else float('inf'))
        achieved_milestones = checkpoint.get('achieved_milestones', set())

    # 读取 MCMC 历史数据
    history_acc, history_phi_means, history_phi_errs, mcmc_steps = [], [], [], []
    if os.path.exists(CONFIG['observables_save_path']):
        try:
            npz_data = np.load(CONFIG['observables_save_path'])
            steps_array = npz_data['steps']
            valid_idx = steps_array < start_iteration
            mcmc_steps = steps_array[valid_idx].tolist()
            history_acc = npz_data['acc'][valid_idx].tolist()
            history_phi_means = npz_data['phi_means'][valid_idx].tolist()
            history_phi_errs = npz_data['phi_errs'][valid_idx].tolist()
            print(f"✅ 成功加载外部物理观测记录，已对齐至第 {start_iteration - 1} 步。")
        except Exception as e:
            print(f"⚠️ 无法读取 {CONFIG['observables_save_path']}，错误: {e}")

    scheduler = None
    if CONFIG.get('use_scheduler', True):
        safe_last_epoch = -1
        if start_iteration > 1:
            safe_last_epoch = min(start_iteration - 1, CONFIG['scheduler_steps'])

            # 👇 🌟 【物理超度：纯数学手算】
            # 不猜 PyTorch 的状态机了，直接根据 Cosine Annealing 公式精准计算！
            import math
            eta_max = CONFIG['lr']
            eta_min = CONFIG['scheduler_min']
            T_max = CONFIG['scheduler_steps']
            exact_lr = eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * safe_last_epoch / T_max))

            # 暴力把算出来的真实 LR 直接灌进优化器
            for param_group in optimizer.param_groups:
                param_group['lr'] = exact_lr

        # 再挂上 Scheduler，传入断点步数。此时它已经被我们的数值架空了，只能乖乖从这步往下走
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['scheduler_steps'],
            eta_min=CONFIG['scheduler_min'],
            last_epoch=safe_last_epoch
        )

        if start_iteration > 1:
            # 这次打印的是我们亲手注入的内存真实值，绝无可能出错
            print(
                f"🔧 [绝对掌控] 彻底绕开 PyTorch 黑盒！已手算强注第 {safe_last_epoch} 步的正确学习率: {optimizer.param_groups[0]['lr']:.2e}")

    if hasattr(torch, 'compile') and device.type == 'cuda':
        model.train()
        compiled_model = torch.compile(model)
        # compiled_model = model
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

        # 🌟 新增：动态判断当前是否需要开启强制对称性
        enforce_sym = CONFIG.get('enforce_z2_sym', False) and (iteration >= CONFIG.get('sym_start_iter', 0))

        # 🌟 修改：将 enforce_sym 传给模型
        phi, log_det_J = compiled_model(z, progress_tensor, enforce_sym=enforce_sym)
        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi))
        loss.backward()

        # 🌟 修改点：方案 A - 全局宏观磁化率惩罚 (Global Magnetization Penalty)
        # 计算整个 Batch 内所有样本、所有格点的平均场值，并惩罚其平方
        # 🌟 修正：补偿体积因子，对齐 Action 的广延量级
        V = CONFIG['L'] * CONFIG['L']
        batch_mag = torch.mean(phi)
        loss_sym = V * (batch_mag ** 2)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            base_name = CONFIG['base_name']
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

        if iteration >= 10000 and iteration % 2000 == 0:
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

    torch.backends.cudnn.benchmark = True

    final_model = train()