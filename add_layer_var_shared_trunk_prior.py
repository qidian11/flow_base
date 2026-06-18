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
    'type': 'add_shared_trunk_prior_cnn',
    'L': 14,
    'm_sq': -4.0,
    'lam': 5.113,
    'batch_size': 512,
    'lr': 1e-4,
    'use_scheduler': True,
    'scheduler_min': 1e-5,
    # 🌟 注意：如果你旧模型已经跑到 36000 步，这里最好改成 60000 甚至 80000 以便让新层有空间训练
    'iterations': 90000,
    'scheduler_steps': 20000,
    'warmup_steps': 10000.0,
    'target_acc_ratio': 0.78,

    'enforce_z2_sym': False,
    'sym_start_iter': 30000,

    # 🌟 [核心控制台]：层扩展与继承开关
    'enable_layer_expansion': True,
    'extend_from_checkpoint': "latest_shared_trunk_prior_cnn_dp_False_L14_c6_d2_TrCh128x6_Ly3x6_trk_3_dil_1_Sh96x6L1x6_Th64x6L1x6_iter_60000.pt",  # 👈 在这里填入你要继承的旧模型绝对或相对路径

    # 🌟 [自动状态机]：不要手动改！代码会通过正则和存档自动填入这里
    'expansion_start_iter': 0,
    'split_lr_at_layer': 0,
    # 👇 🌟 新增：新层独立预热多少步之后，全网解冻协同训练？
    'unfreeze_old_layers_after': 12000,

    # 🌟 扩展为 8 层
    # 'cnn_coupling_layers': 8, 改为读取trunk_channels长度
    'branch_depth': 2,
    'double_precision': False,

    # 🌟 列表长度也全部补齐为 8 个元素
    'trunk_channels': [128, 128, 128, 128, 128, 128, 128, 128],
    'trunk_layers': [3, 3, 3, 3, 3, 3, 3, 3],
    'trunk_use_multi_kernel': True,
    'trunk_kernel_sizes': (3,),
    'trunk_dilations': (1,),

    's_head_channels': [96, 96, 96, 96, 96, 96, 96, 96],
    's_head_layers': [1, 1, 1, 1, 1, 1, 1, 1],
    's_head_kernel_size': 3,

    't_head_channels': [64, 64, 64, 64, 64, 64, 64, 64],
    't_head_layers': [1, 1, 1, 1, 1, 1, 1, 1],
    't_kernel_size': 3,
}

# ==========================================
# 🌟 [动态配置计算与安全断言]
# ==========================================
# 1. 强制 Single Source of Truth：层数由主干通道列表长度唯一决定
CONFIG['cnn_coupling_layers'] = len(CONFIG['trunk_channels'])

# 2. 防御性编程：严格检查所有涉及层数的列表，只要长度对不上，程序立刻在启动时自爆报错
_list_keys_to_check = ['trunk_layers', 's_head_channels', 's_head_layers', 't_head_channels', 't_head_layers']
for _k in _list_keys_to_check:
    if _k in CONFIG and isinstance(CONFIG[_k], list):
        assert len(CONFIG[_k]) == CONFIG['cnn_coupling_layers'], \
            f"❌ 配置致命错误: 列表 '{_k}' 的长度 ({len(CONFIG[_k])}) 与 'trunk_channels' ({CONFIG['cnn_coupling_layers']}) 不一致！"

tk_str = '_'.join(map(str, CONFIG.get('trunk_kernel_sizes', (3, 3))))
tdil_str = '_'.join(map(str, CONFIG.get('trunk_dilations', (1, 2))))
trunk_rf = f"trk_{tk_str}_dil_{tdil_str}"

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

    # 1. 加上 warmup_start_idx 参数，默认从第 0 层开始预热
    def forward(self, z, progress=None, enforce_sym=False, warmup_start_idx=0):
        phi = z
        log_det_jacobian = 0

        # 2. 这里的循环加上 enumerate 获取当前层索引 i
        for i, net in enumerate(self.context_nets):
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

                # 👇 🌟 核心修复：只有当当前层索引 i 大于等于 warmup_start_idx 时，才应用截断！
                if self.training and progress is not None and i >= warmup_start_idx:
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
# ==========================================
# 7. 训练主循环 (🌟 兼容层堆叠的热插拔设计)
# ==========================================
# ==========================================
# 7. 训练主循环 (🌟 全自动状态机、正则解析、热插拔支持)
# ==========================================
def train():
    import re  # 引入正则匹配模块

    model = FlowModel(CONFIG).to(device)
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=abs(CONFIG['m_sq'])).to(device)

    if CONFIG.get('double_precision', False):
        model = model.double()
        prior = prior.double()

    eps_val = 1e-15 if CONFIG.get('double_precision', False) else 1e-8
    train_dtype = torch.float64 if CONFIG.get('double_precision', False) else torch.float32

    history_loss = []
    best_loss = float('inf')
    ema_loss = None
    start_iteration = 1
    target_milestones = sorted([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75], reverse=True)
    achieved_milestones = set()

    # 1. 获取要加载的存档路径 (全量从 CONFIG 获取)
    old_checkpoint_path = auto_find_latest_checkpoint(CONFIG)
    if not old_checkpoint_path and CONFIG.get('extend_from_checkpoint') and os.path.exists(
            CONFIG['extend_from_checkpoint']):
        old_checkpoint_path = CONFIG['extend_from_checkpoint']
        print(f"\n🔄 [加载入口]：正尝试从指定旧模型 [{old_checkpoint_path}] 继承参数...")

    is_expanding_layers = False
    loaded_layer_count = CONFIG['cnn_coupling_layers']

    # ========================== [自动状态机：读取与对齐] ==========================
    if old_checkpoint_path and os.path.exists(old_checkpoint_path):
        print(f"\n从 {old_checkpoint_path} 读取并初始化网络...")
        checkpoint = torch.load(old_checkpoint_path, map_location=device)

        # 尝试从存档中直接恢复扩层记忆 (解决中断后重新执行的问题)
        if 'expansion_start_iter' in checkpoint: CONFIG['expansion_start_iter'] = checkpoint['expansion_start_iter']
        if 'split_lr_at_layer' in checkpoint: CONFIG['split_lr_at_layer'] = checkpoint['split_lr_at_layer']

        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        old_layer_indices = [int(k.split('.')[1]) for k in clean_dict.keys() if k.startswith('context_nets.')]
        loaded_layer_count = max(old_layer_indices) + 1 if old_layer_indices else 0

        current_dict = model.state_dict()
        filtered_dict = {k: v for k, v in clean_dict.items() if k in current_dict and v.shape == current_dict[k].shape}
        model.load_state_dict(filtered_dict, strict=False)

        # 核心判断：是否触发新增层生长模式
        if CONFIG.get('enable_layer_expansion', False) and (CONFIG['cnn_coupling_layers'] > loaded_layer_count):
            is_expanding_layers = True

            # 神来之笔：利用正则自动从文件名提取最后步数，作为新层的原点
            match = re.search(r'iter_(\d+)', old_checkpoint_path)
            if match:
                CONFIG['expansion_start_iter'] = int(match.group(1))
                print(f"🕵️ [状态机] 自动探测：从文件名提取旧模型最终步数 -> {CONFIG['expansion_start_iter']}")
            else:
                CONFIG['expansion_start_iter'] = checkpoint.get('iteration', 0)

            CONFIG['split_lr_at_layer'] = loaded_layer_count

            print(f"✨✨✨ 架构扩展模式已激活 ✨✨✨")
            print(
                f"✅ 检测到旧模型为 {loaded_layer_count} 层，当前配置为 {CONFIG['cnn_coupling_layers']} 层。已执行恒等映射！")
            start_iteration = checkpoint['iteration'] + 1
        else:
            start_iteration = checkpoint['iteration'] + 1
            print(f"✅ 正常恢复模型训练，当前层数: {CONFIG['cnn_coupling_layers']}")

        history_loss = checkpoint.get('history_loss', [])
        ema_loss = checkpoint.get('ema_loss', None)
        best_loss = checkpoint.get('best_loss', min(history_loss) if history_loss else float('inf'))
        achieved_milestones = checkpoint.get('achieved_milestones', set())

    # ========================== [动态分配优化器] ==========================
    split_layer = CONFIG.get('split_lr_at_layer', 0)
    if split_layer > 0 and split_layer < CONFIG['cnn_coupling_layers']:
        old_params, new_params = [], []
        for i, net in enumerate(model.context_nets):
            if i < split_layer:
                old_params.extend(list(net.parameters()))
            else:
                new_params.extend(list(net.parameters()))

        optimizer = optim.Adam([
            {'params': old_params, 'lr': 1e-5},
            {'params': new_params, 'lr': CONFIG['lr']}
        ], eps=eps_val)
        print(f"🔧 [优化器] 已在第 {split_layer} 层处隔离新旧层学习率 (旧层 1e-5, 新层 {CONFIG['lr']})。")
    else:
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], eps=eps_val)

    if old_checkpoint_path and os.path.exists(old_checkpoint_path):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError:
            print(f"⚠️ [警告] 优化器组结构变动！已自动重置优化器状态 (切换学习率策略时的正常安全现象)。")

        if CONFIG.get('double_precision', False):
            for group in optimizer.param_groups: group['eps'] = 1e-15
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point(): state[k] = v.double()

    # ========================== [调度器初始化] ==========================
    scheduler = None
    if CONFIG.get('use_scheduler', True):
        exp_start = CONFIG.get('expansion_start_iter', 0)
        if exp_start > 0:
            remaining_steps = CONFIG['iterations'] - exp_start
            current_step = start_iteration - exp_start
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_steps, eta_min=CONFIG['scheduler_min'],
                last_epoch=current_step - 1 if current_step > 1 else -1
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CONFIG['scheduler_steps'], eta_min=CONFIG['scheduler_min'],
                last_epoch=start_iteration - 1 if start_iteration > 1 else -1
            )

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

    compiled_model = model
    model.train()

    # ========================== [训练主循环] ==========================
    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        # 👇 🌟 新增：全网自动解冻逻辑
        exp_start = CONFIG.get('expansion_start_iter', 0)
        unfreeze_delay = CONFIG.get('unfreeze_old_layers_after', 0)

        if exp_start > 0 and unfreeze_delay > 0:
            is_past_unfreeze = (iteration >= exp_start + unfreeze_delay)
            # 如果到达了解冻节点，且检测到旧层学习率(组0)依然被压制在极小值，则立即对齐学习率
            # 直接判断调度器的基准峰值是否尚未对齐，彻底免疫余弦曲线的当前位置干扰
            if is_past_unfreeze and scheduler and scheduler.base_lrs[0] < scheduler.base_lrs[1]:
                print(f"\n🔓 [全网解冻] 触发！新层已完成 {unfreeze_delay} 步独立预热。")
                print(f"🚀 正在对齐新旧层学习率，开启全网协同联合优化...")

                # 强行将旧层的学习率拉上来，与新层同步
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[1]['lr']

                # 必须同步更新调度器底座，否则下一步又会被拉回去
                if scheduler:
                    scheduler.base_lrs[0] = scheduler.base_lrs[1]
        # 👆 ==========================================

        optimizer.zero_grad()
        z, log_p_z = prior.sample(CONFIG['batch_size'])

        # 🌟 根据状态机严谨计算 Warmup
        exp_start = CONFIG.get('expansion_start_iter', 0)
        if exp_start > 0:
            steps_since_restart = iteration - exp_start
            progress_val = min(max(steps_since_restart, 0) / CONFIG['warmup_steps'], 1.0)
        else:
            progress_val = min(iteration / CONFIG['warmup_steps'], 1.0)

        model.step_warmup(progress_val)
        enforce_sym = CONFIG.get('enforce_z2_sym', False) and (iteration >= CONFIG.get('sym_start_iter', 0))

        # 🌟 将边界保护精准锁定在扩层的起始层
        start_idx_for_warmup = CONFIG.get('split_lr_at_layer', 0)

        phi, log_det_J = compiled_model(z,
                                        torch.tensor(progress_val, device=device, dtype=train_dtype),
                                        enforce_sym=enforce_sym,
                                        warmup_start_idx=start_idx_for_warmup)

        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi))
        loss.backward()
        loss_sym = (CONFIG['L'] ** 2) * (torch.mean(phi) ** 2)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            optimizer.zero_grad()
            continue
        else:
            optimizer.step()

        if scheduler:
            if exp_start > 0:
                scheduler.step()  # 扩展模式下全局步进
            elif iteration <= CONFIG['scheduler_steps']:
                scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        # ========================== [状态机记忆保存机制] ==========================
        checkpoint_dict = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss_val, 'best_loss': best_loss,
            'history_loss': history_loss, 'ema_loss': ema_loss,
            'achieved_milestones': achieved_milestones,
            'expansion_start_iter': CONFIG.get('expansion_start_iter', 0),  # 👈 将记忆写入物理存档
            'split_lr_at_layer': CONFIG.get('split_lr_at_layer', 0)  # 👈 将记忆写入物理存档
        }

        if iteration % 100 == 0:
            current_lr = optimizer.param_groups[-1]['lr']
            print(f"迭代 {iteration:6d}/{CONFIG['iterations']} | Z_2: {'ON' if enforce_sym else 'OFF'} "
                  f"| 瞬时: {loss_val:.4f} | 平滑: {ema_loss:.4f} | Best: {best_loss:.4f} | sym: {loss_sym:.4f} | LR: {current_lr:.2e}")
            torch.save(checkpoint_dict, CONFIG['checkpoint_path'])

        if ema_loss < best_loss and iteration >= 10000:
            best_loss = ema_loss
            checkpoint_dict['best_loss'] = best_loss
            torch.save(checkpoint_dict, CONFIG['save_path'])

        if iteration % 2000 == 0:
            torch.save(checkpoint_dict, f"iter_{iteration}_{CONFIG['base_name']}.pt")

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
                    base_ckpt = CONFIG['checkpoint_path'].replace('latest_', '')
                    torch.save(checkpoint_dict, f"acc_{int(m * 100)}percent_{base_ckpt}")
                    break

    print("✅ 训练流水线结束！")
    return model


if __name__ == "__main__":
    # 👇 🌟 新增：强制清理次正常数，防止 GPU 遇到梯度爆炸时软挂起卡死
    torch.set_flush_denormal(True)

    if CONFIG.get('double_precision', False): torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.benchmark = True
    final_model = train()