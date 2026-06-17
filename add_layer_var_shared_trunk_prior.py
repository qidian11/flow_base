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

# ==========================================
# 🌟 [关键入口]：跨代继承权重路径
# ==========================================
# 如果你要把 6 层扩展为 8 层，请把这里填上你 6 层模型 best_ 或 latest_ 的真实绝对/相对路径！
# 例如："iter_36000_shared_trunk_prior_cnn_dp_False_L14_c6_...pt"
# 如果留空，代码会认为你想从零开始训练全新的 8 层网络。
EXTEND_FROM_CHECKPOINT = ""


def is_valid_checkpoint(filepath):
    try:
        torch.load(filepath, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def fmt_cfg(val):
    if not isinstance(val, (list, tuple)): return str(val)
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
    for file in get_sorted_files("iter_*_"):
        if is_valid_checkpoint(file): return file

    for file in get_sorted_files(""):
        filename = os.path.basename(file)
        if not filename.startswith(("latest_", "best_", "iter_")) and is_valid_checkpoint(file):
            return file
    return None


# ==========================================
# 1. 物理参数配置 (已变更为 8 层)
# ==========================================
CONFIG = {
    'type': 'shared_trunk_prior_cnn',
    'L': 14,
    'm_sq': -4.0,
    'lam': 5.113,
    'batch_size': 512,
    'lr': 1e-3,
    'use_scheduler': True,
    'scheduler_min': 1e-5,
    # 🌟 注意：如果你旧模型已经跑到 36000 步，这里最好改成 60000 甚至 80000 以便让新层有空间训练
    'iterations': 60000,
    'scheduler_steps': 35000,
    'warmup_steps': 3000.0,
    'target_acc_ratio': 0.78,

    'enforce_z2_sym': False,
    'sym_start_iter': 30000,

    # 🌟 扩展为 8 层
    'cnn_coupling_layers': 8,
    'branch_depth': 2,
    'double_precision': False,

    # 🌟 列表长度也全部补齐为 8 个元素
    'trunk_channels': [24, 24, 24, 24, 24, 24, 24, 24],
    'trunk_layers': [3, 3, 3, 3, 3, 3, 3, 3],
    'trunk_use_multi_kernel': True,
    'trunk_kernel_sizes': (3,),
    'trunk_dilations': (1,),

    's_head_channels': [24, 24, 24, 24, 24, 24, 24, 24],
    's_head_layers': [0, 0, 0, 0, 0, 0, 0, 0],
    's_head_kernel_size': 3,

    't_head_channels': [24, 24, 24, 24, 24, 24, 24, 24],
    't_head_layers': [0, 0, 0, 0, 0, 0, 0, 0],
    't_kernel_size': 3,
}

base_name = (f"{CONFIG.get('type', 'shared_trunk_prior_cnn')}_dp_{CONFIG.get('double_precision', True)}_"
             f"L{CONFIG['L']}_c{CONFIG['cnn_coupling_layers']}_d{CONFIG.get('branch_depth', 2)}_"
             f"TrCh{fmt_cfg(CONFIG.get('trunk_channels'))}_Ly{fmt_cfg(CONFIG.get('trunk_layers'))}_{trunk_rf}_"
             f"Sh{fmt_cfg(CONFIG.get('s_head_channels'))}L{fmt_cfg(CONFIG.get('s_head_layers'))}_"
             f"Th{fmt_cfg(CONFIG.get('t_head_channels'))}L{fmt_cfg(CONFIG.get('t_head_layers'))}_"
             f"iter_{CONFIG['iterations']}")

CONFIG['base_name'] = base_name
save_path = f"best_{base_name}.pt"
loss_save_path = f"loss_{base_name}.npy"
checkpoint_path = f"latest_{base_name}.pt"
phi_ensemble_save_path = f"phi_ensemble_{base_name}.npz"
CONFIG['save_path'] = save_path
CONFIG['loss_save_path'] = loss_save_path
CONFIG['checkpoint_path'] = checkpoint_path
CONFIG['observables_save_path'] = f"observables_{base_name}.npz"
CONFIG['phi_ensemble_save_path'] = phi_ensemble_save_path

# ==========================================
# 2. 标量场理论的 Action 计算 (🌟 移除死绑的 dtype，实现精度动态自适应)
# ==========================================
laplacian_kernel_base = torch.tensor([[
    [0.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 0.0]
]], device=device)


def compute_action(phi):
    phi_padded = F.pad(phi, pad=(1, 1, 1, 1), mode='circular')
    # 🌟 动态转为当前 phi 的精度 (FP32/FP64)
    lap_k = laplacian_kernel_base.to(dtype=phi.dtype, device=phi.device).unsqueeze(1)
    laplacian = F.conv2d(phi_padded, lap_k)
    action_density = phi * laplacian + CONFIG['m_sq'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 掩码生成与卷积上下文网络
# ==========================================
def create_checkerboard_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, branch_depth=3):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        for _ in range(branch_depth):
            layers.append(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=pad, padding_mode='circular'))
            layers.append(nn.LeakyReLU(0.01))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), branch_depth=3):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            pad_size = d * (k - 1) // 2
            layers = nn.ModuleList()
            for _ in range(branch_depth):
                layers.append(
                    nn.Conv2d(channels, channels, kernel_size=k, dilation=d, padding=pad_size, padding_mode='circular',
                              bias=False))
                layers.append(nn.LeakyReLU(0.01))
            self.branches.append(layers)
        self.fusion_conv = nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False)

    def forward(self, x):
        outs = []
        for branch in self.branches:
            out = x
            for i in range(0, len(branch), 2):
                conv_layer = branch[i]
                act_layer = branch[i + 1]
                out = act_layer(conv_layer(out))
            outs.append(out)
        fused = torch.cat(outs, dim=1)
        out = self.fusion_conv(fused)
        return x + out


class ConvContextNet(nn.Module):
    def __init__(self, trunk_channels=64, trunk_layers=3, trunk_use_multi_kernel=True, trunk_kernel_sizes=(3, 3),
                 trunk_dilations=(1, 2),
                 s_head_channels=32, s_head_layers=1, s_head_kernel_size=3, t_head_channels=32, t_head_layers=1,
                 t_kernel_size=3, branch_depth=2):
        super().__init__()
        self.s_head_layers = s_head_layers
        self.t_head_layers = t_head_layers

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

        s_pad = s_head_kernel_size // 2
        if s_head_layers == 0:
            self.s_head = nn.Conv2d(trunk_channels, 1, kernel_size=s_head_kernel_size, padding=s_pad,
                                    padding_mode='circular')
        else:
            s_head_list = [nn.Conv2d(trunk_channels, s_head_channels, kernel_size=s_head_kernel_size, padding=s_pad,
                                     padding_mode='circular'), nn.LeakyReLU(0.01)]
            for _ in range(s_head_layers):
                s_head_list.append(ResBlock(s_head_channels, kernel_size=s_head_kernel_size, branch_depth=branch_depth))
            s_head_list.append(nn.Conv2d(s_head_channels, 1, kernel_size=1))
            self.s_head = nn.Sequential(*s_head_list)

        t_pad = t_kernel_size // 2
        if t_head_layers == 0:
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
        if self.s_head_layers == 0:
            nn.init.zeros_(self.s_head.weight)
            nn.init.zeros_(self.s_head.bias)
        else:
            nn.init.normal_(self.s_head[0].weight, mean=0, std=0.05)
            nn.init.zeros_(self.s_head[-1].weight)
            nn.init.zeros_(self.s_head[-1].bias)

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
        dtype = torch.float64 if config.get('double_precision', False) else torch.float32

        self.register_buffer('s_bounds', torch.tensor([-0.5, 4.0], dtype=dtype))
        self.register_buffer('t_bounds', torch.tensor([-15.0, 15.0], dtype=dtype))

        def get_layer_cfg(key, default, layer_idx):
            val = config.get(key, default)
            if isinstance(val, (list, tuple)):
                assert len(val) == self.cnn_layers, f"❌ 参数 {key} 长度必须等于 cnn_coupling_layers"
                return val[layer_idx]
            return val

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

        print("=" * 70)
        print(f"🌟 Variable Shared-Trunk 模型初始化完毕！当前层数: {self.total_layers}")
        print("=" * 70)

    def step_warmup(self, progress):
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

    def forward(self, z, progress=None, enforce_sym=False):
        phi = z
        log_det_jacobian = 0

        for net in self.context_nets:
            for step in range(2):
                current_mask = self.base_mask if step == 0 else (1.0 - self.base_mask)
                phi_frozen = current_mask * phi

                if enforce_sym:
                    st_out_pos = net(phi_frozen)
                    st_out_neg = net(-phi_frozen)
                    s_out_pos, t_out_pos = st_out_pos[:, 0:1, :, :], st_out_pos[:, 1:2, :, :]
                    s_out_neg, t_out_neg = st_out_neg[:, 0:1, :, :], st_out_neg[:, 1:2, :, :]
                    s_out = (s_out_pos + s_out_neg) / 2.0
                    t_out = (t_out_pos - t_out_neg) / 2.0
                else:
                    st_out = net(phi_frozen)
                    s_out, t_out = st_out[:, 0:1, :, :], st_out[:, 1:2, :, :]

                if self.training and progress is not None:
                    is_warmup = (progress < 1.0).view(1, 1, 1, 1)
                    s_clamped = asymmetric_soft_clamp(s_out, self.s_bounds[0], self.s_bounds[1])
                    t_clamped = torch.clamp(t_out, self.t_bounds[0], self.t_bounds[1])
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


# ==========================================
# 6. 一体化评估核心 (🌟 强制双精度保护)
# ==========================================
def run_mcmc_evaluation(model, prior, total_n=10000, batch_size=1024, enforce_sym=False):
    model.eval()

    # 🌟 核心保护：动态强制提升至 FP64，避免大网格累加截断误差
    original_dtype = next(model.parameters()).dtype
    model = model.double()
    prior = prior.double()
    eval_dtype = torch.float64

    total_n = (total_n // batch_size) * batch_size

    # 使用高精度分配内存
    all_phis = torch.empty((total_n, 1, CONFIG['L'], CONFIG['L']), dtype=eval_dtype, device=device)
    all_s = torch.empty(total_n, dtype=eval_dtype, device=device)
    all_log_qs = torch.empty(total_n, dtype=eval_dtype, device=device)

    dummy_progress = torch.tensor(1.0, device=device, dtype=eval_dtype)

    with torch.no_grad():
        for i in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - i)
            z, log_p_z = prior.sample(current_batch)
            phi, log_det_J = model(z, dummy_progress, enforce_sym=enforce_sym)

            all_phis[i:i + current_batch] = phi
            all_s[i:i + current_batch] = compute_action(phi)
            all_log_qs[i:i + current_batch] = log_p_z - log_det_J

    accepted_count = 0
    curr_phi = all_phis[0]
    curr_s = all_s[0]
    curr_log_q = all_log_qs[0]

    log_rands = torch.log(torch.rand(total_n, device=device, dtype=eval_dtype))
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

        chain_phis[i] = curr_phi

    phi_powers_mean = []
    phi_powers_err = []
    for power in range(1, 6):
        pow_seq = (chain_phis ** power).mean(dim=(2, 3)).squeeze()
        phi_powers_mean.append(pow_seq.mean().item())
        phi_powers_err.append((pow_seq.std() / math.sqrt(total_n)).item())

    # 🌟 评估结束，恢复至原训练精度
    if original_dtype == torch.float32:
        model = model.float()
        prior = prior.float()
    model.train()

    return accepted_count / (total_n - 1), phi_powers_mean, phi_powers_err


# ==========================================
# 7. 训练主循环 (🌟 兼容层堆叠的热插拔设计)
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

    # 1. 尝试寻找当前 8 层架构的文件
    old_checkpoint_path = auto_find_latest_checkpoint(CONFIG)

    # 2. 如果没找到，但用户填了 EXTEND_FROM_CHECKPOINT，则尝试从 6 层存档继承！
    if not old_checkpoint_path and EXTEND_FROM_CHECKPOINT and os.path.exists(EXTEND_FROM_CHECKPOINT):
        old_checkpoint_path = EXTEND_FROM_CHECKPOINT
        print(f"\n⚠️ 未找到当前 {CONFIG['cnn_coupling_layers']} 层的匹配存档！")
        print(f"🔄 [渐进式层堆叠激活]：正尝试从旧模型 [{EXTEND_FROM_CHECKPOINT}] 继承参数...")

    if old_checkpoint_path and os.path.exists(old_checkpoint_path):
        print(f"\n从 {old_checkpoint_path} 读取并初始化网络...")
        checkpoint = torch.load(old_checkpoint_path, map_location=device)
        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}

        # 🌟 核心兼容加载逻辑：只加载矩阵形状和名字严丝合缝对齐的部分
        current_dict = model.state_dict()
        filtered_dict = {k: v for k, v in clean_dict.items() if k in current_dict and v.shape == current_dict[k].shape}
        model.load_state_dict(filtered_dict, strict=False)

        # 🌟 探测是否发生了架构拓展
        if len(filtered_dict) < len(current_dict):
            print(f"✨✨✨ 架构扩展成功 ✨✨✨")
            print(f"✅ 从旧模型中成功复用 {len(filtered_dict)} 个参数张量。")
            print(f"✅ 新增的顶层网络已被自动执行 [恒等映射] (输出 s=0, t=0)。")
            print(f"⚠️ 优化器状态已重置以容纳新层，但总体训练步数将延续。")
            # 👇 🌟 新增：动态冻结旧层 (前 6 层)
            old_layer_count = 6  # 你继承的旧模型层数
            for i in range(old_layer_count):
                for param in model.context_nets[i].parameters():
                    param.requires_grad = False
            print(f"❄️ [热身模式激活]：已物理冻结前 {old_layer_count} 层的梯度，初期只训练新增层！")

            start_iteration = checkpoint['iteration'] + 1
            # 不加载旧优化器，让新优化器接管
        else:
            # 架构没变，正常恢复状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if CONFIG.get('double_precision', False):
                for group in optimizer.param_groups: group['eps'] = 1e-15
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.is_floating_point(): state[k] = v.double()
            start_iteration = checkpoint['iteration'] + 1

        # 🌟 无缝补偿：确保微调新层时的学习率是保守的底线，防止冲垮旧层
        if start_iteration > CONFIG['scheduler_steps']:
            for param_group in optimizer.param_groups:
                param_group['lr'] = CONFIG['scheduler_min']
            print(f"🔧 [微调锁死]：已超过退火期，无缝锁定最低学习率: {CONFIG['scheduler_min']:.2e}")

        history_loss = checkpoint.get('history_loss', [])
        ema_loss = checkpoint.get('ema_loss', None)
        best_loss = checkpoint.get('best_loss', min(history_loss) if history_loss else float('inf'))
        achieved_milestones = checkpoint.get('achieved_milestones', set())

    history_acc, history_phi_means, history_phi_errs, mcmc_steps = [], [], [], []
    if os.path.exists(CONFIG['observables_save_path']):
        try:
            npz_data = np.load(CONFIG['observables_save_path'])
            valid_idx = npz_data['steps'] < start_iteration
            mcmc_steps = npz_data['steps'][valid_idx].tolist()
            history_acc = npz_data['acc'][valid_idx].tolist()
            history_phi_means = npz_data['phi_means'][valid_idx].tolist()
            history_phi_errs = npz_data['phi_errs'][valid_idx].tolist()
        except Exception:
            pass

    scheduler = None
    if CONFIG.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['scheduler_steps'], eta_min=CONFIG['scheduler_min'],
            last_epoch=start_iteration - 1 if start_iteration > 1 else -1
        )

    compiled_model = model
    model.train()

    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        # 👇 🌟 新增：两阶段微调定时器 (例如在热身 2000 步后解冻)
        warmup_train_steps = 2000
        if iteration == start_iteration + warmup_train_steps:
            for param in model.parameters():
                param.requires_grad = True
            print(f"\n{'=' * 50}")
            print(f"🔥 [全网解冻]：热身结束！已重新开启所有层的梯度，进入全网联合微调阶段！")
            print(f"{'=' * 50}\n")

        optimizer.zero_grad()
        z, log_p_z = prior.sample(CONFIG['batch_size'])

        progress_val = min(iteration / CONFIG['warmup_steps'], 1.0)
        model.step_warmup(progress_val)

        enforce_sym = CONFIG.get('enforce_z2_sym', False) and (iteration >= CONFIG.get('sym_start_iter', 0))
        phi, log_det_J = compiled_model(z, torch.tensor(progress_val, device=device, dtype=dtype),
                                        enforce_sym=enforce_sym)

        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi))
        loss.backward()

        loss_sym = (CONFIG['L'] ** 2) * (torch.mean(phi) ** 2)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            optimizer.zero_grad()
            continue
        else:
            optimizer.step()

        if scheduler and iteration <= CONFIG['scheduler_steps']:
            scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        if iteration % 100 == 0:
            print(f"迭代 {iteration:6d}/{CONFIG['iterations']} | Z_2: {'ON' if enforce_sym else 'OFF'} "
                  f"| 瞬时: {loss_val:.4f} | 平滑: {ema_loss:.4f} | Best: {best_loss:.4f} | sym: {loss_sym:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss, 'ema_loss': ema_loss,
                        'achieved_milestones': achieved_milestones}, checkpoint_path)

        if ema_loss < best_loss and iteration >= 10000:
            best_loss = ema_loss
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss_val, 'best_loss': best_loss, 'history_loss': history_loss, 'ema_loss': ema_loss,
                        'achieved_milestones': achieved_milestones}, save_path)

        if iteration % 2000 == 0:
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': loss_val, 'best_loss': best_loss},
                       f"iter_{iteration}_{CONFIG['base_name']}.pt")

        if iteration >= 10000 and iteration % 5000 == 0:
            total_n = 50000 if iteration % 5000 == 0 else 100000
            print(f"\n{'=' * 50}\n🚀 [迭代 {iteration}] 触发 MCMC 在线验证 (样本量: {total_n})...")

            acc_rate, phi_means, phi_errs = run_mcmc_evaluation(compiled_model, prior, total_n=total_n,
                                                                batch_size=CONFIG['batch_size'],
                                                                enforce_sym=enforce_sym)

            print(f"📊 当前物理接受率: {acc_rate:.2%}")
            for p in range(1, 6): print(f"phi^{p}: {phi_means[p - 1]:.6f} ± {phi_errs[p - 1]:.6f}")
            print(f"{'=' * 50}\n")

            mcmc_steps.append(iteration)
            history_acc.append(acc_rate)
            history_phi_means.append(phi_means)
            history_phi_errs.append(phi_errs)
            np.savez(CONFIG['observables_save_path'], steps=np.array(mcmc_steps), acc=np.array(history_acc),
                     phi_means=np.array(history_phi_means), phi_errs=np.array(history_phi_errs))

            for m in target_milestones:
                if acc_rate >= m and m not in achieved_milestones:
                    print(f"⭐ 达成里程碑！接受率突破 {m:.0%}，正在保存专属模型...")
                    achieved_milestones.update([lm for lm in target_milestones if lm <= m])
                    torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               f"acc_{int(m * 100)}percent_{checkpoint_path.replace('latest_', '')}")
                    break

    print("✅ 训练流水线结束！")
    return model


if __name__ == "__main__":
    if CONFIG.get('double_precision', False): torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.benchmark = True
    final_model = train()