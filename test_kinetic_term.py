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


# ==========================================
# 1. 动态命名与断点搜索工具
# ==========================================
def is_valid_checkpoint(filepath):
    try:
        torch.load(filepath, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def get_base_name(config):
    """直接以实验名称和核心参数作为文件标识，确保完全隔离"""
    base_name = (
        f"{config['exp_name']}_dp_{config.get('double_precision', False)}_"
        f"L{config['L']}_c{config['cnn_coupling_layers']}_iter_{config['iterations']}"
    )
    return base_name


def auto_find_latest_checkpoint(config):
    base_name = get_base_name(config)
    base_pattern = base_name.replace(f"iter_{config['iterations']}", "iter_*") + ".pt"

    def get_sorted_files(prefix):
        files = glob.glob(prefix + base_pattern)
        return sorted(files, key=os.path.getmtime, reverse=True)

    for file in get_sorted_files("latest_"):
        if is_valid_checkpoint(file): return file
    for file in get_sorted_files("best_"):
        if is_valid_checkpoint(file): return file
    for file in get_sorted_files("iter_*_"):
        if is_valid_checkpoint(file): return file
    return None


# ==========================================
# 2. 标量场理论的 Action 计算 (支持实验切换)
# ==========================================
laplacian_kernel = torch.tensor([[
    [0.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 0.0]
]], device=device).unsqueeze(1)


def compute_action(phi, config):
    """根据实验配置动态切换物理项"""
    action_density = torch.zeros_like(phi)

    # 动能项 (空间关联)
    if config.get('use_kinetic', True):
        phi_padded = F.pad(phi, pad=(1, 1, 1, 1), mode='circular')
        adaptive_kernel = laplacian_kernel.to(dtype=phi.dtype, device=phi.device)
        laplacian = F.conv2d(phi_padded, adaptive_kernel)
        action_density += phi * laplacian

    # 质量项 (始终保留以保证积分收敛)
    action_density += config['m_sq'] * (phi ** 2)

    # 势能相互作用项 (局域非线性)
    if config.get('use_potential', True):
        action_density += config['lam'] * (phi ** 4)

    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 经典棋盘掩码生成与卷积上下文网络
# ==========================================
def create_checkerboard_mask(L):
    """生成严格的 1x1 棋盘掩码 (Checkerboard Mask)"""
    indices = torch.arange(L)
    x, y = torch.meshgrid(indices, indices, indexing='ij')
    mask_2d = ((x + y) % 2 == 0).float()
    return mask_2d.view(1, 1, L, L)


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
        nn.init.zeros_(self.block[-2].weight)
        if self.block[-2].bias is not None: nn.init.zeros_(self.block[-2].bias)

    def forward(self, x):
        return x + self.block(x)


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), branch_depth=3):
        super().__init__()
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
        nn.init.zeros_(self.fusion_conv.weight)

    def forward(self, x):
        outs = []
        for branch in self.branches:
            out = x
            for i in range(0, len(branch), 2): out = branch[i + 1](branch[i](out))
            outs.append(out)
        return x + self.fusion_conv(torch.cat(outs, dim=1))


class ConvContextNet(nn.Module):
    def __init__(self, trunk_channels=64, trunk_layers=3, trunk_use_multi_kernel=True,
                 trunk_kernel_sizes=(3, 3), trunk_dilations=(1, 2),
                 s_head_channels=32, s_head_layers=1, s_head_kernel_size=3,
                 t_head_channels=32, t_head_layers=1, t_kernel_size=3, branch_depth=2):
        super().__init__()
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

        def build_head(ch, ly, k_size):
            pad = k_size // 2
            if ly == 0: return nn.Conv2d(trunk_channels, 1, kernel_size=k_size, padding=pad, padding_mode='circular')
            head_list = [nn.Conv2d(trunk_channels, ch, kernel_size=k_size, padding=pad, padding_mode='circular'),
                         nn.LeakyReLU(0.01)]
            for _ in range(ly): head_list.append(ResBlock(ch, kernel_size=k_size, branch_depth=branch_depth))
            head_list.append(nn.Conv2d(ch, 1, kernel_size=1))
            return nn.Sequential(*head_list)

        self.s_head = build_head(s_head_channels, s_head_layers, s_head_kernel_size)
        self.t_head = build_head(t_head_channels, t_head_layers, t_kernel_size)

        nn.init.normal_(self.trunk_net[0].weight, mean=0, std=0.1)
        for head, ly in [(self.s_head, s_head_layers), (self.t_head, t_head_layers)]:
            if ly == 0:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            else:
                nn.init.normal_(head[0].weight, mean=0, std=0.05)
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        shared = self.trunk_net(x)
        return torch.cat([self.s_head(shared), self.t_head(shared)], dim=1)


# ==========================================
# 4. 标准高斯先验
# ==========================================
class StandardGaussianPrior(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.register_buffer('const_factor', torch.tensor(0.5 * (L * L) * math.log(2.0 * math.pi)))

    def sample(self, batch_size):
        z = torch.randn(batch_size, 1, self.L, self.L, device=self.const_factor.device, dtype=self.const_factor.dtype)
        log_p_z = -0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - self.const_factor
        return z, log_p_z


# ==========================================
# 5. 流模型
# ==========================================
def asymmetric_soft_clamp(x, min_val, max_val):
    pos_val = max_val * torch.tanh(x / max_val)
    neg_val = abs(min_val) * torch.tanh(x / abs(min_val))
    return torch.where(x >= 0, pos_val, neg_val)


class FlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config['L']
        self.cnn_layers = config['cnn_coupling_layers']

        # 挂载严格的棋盘掩码
        self.register_buffer('base_mask', create_checkerboard_mask(self.L))

        self.context_nets = nn.ModuleList()
        dtype = torch.float64 if config.get('double_precision', False) else torch.float32

        self.register_buffer('s_bounds', torch.tensor([-0.2, 3.0], dtype=dtype))
        self.register_buffer('t_bounds', torch.tensor([-3.0, 3.0], dtype=dtype))

        def get_layer_cfg(key, default, idx):
            val = config.get(key, default)
            return val[idx] if isinstance(val, (list, tuple)) else val

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

    def step_warmup(self, progress):
        clamped = min(progress, 1.0)
        self.s_bounds[0].fill_(-0.2 - 0.3 * clamped)
        self.s_bounds[1].fill_(3.0 + 2.0 * clamped)
        self.t_bounds[0].fill_(-3.0 - 5.0 * clamped)
        self.t_bounds[1].fill_(3.0 + 5.0 * clamped)

    def forward(self, z, progress=None, enforce_sym=False):
        phi, log_det_jacobian = z, 0
        for step, net in enumerate(self.context_nets):
            current_mask = self.base_mask if step % 2 == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi

            if enforce_sym:
                st_out_pos, st_out_neg = net(phi_frozen), net(-phi_frozen)
                s_out = (st_out_pos[:, 0:1] + st_out_neg[:, 0:1]) / 2.0
                t_out = (st_out_pos[:, 1:2] - st_out_neg[:, 1:2]) / 2.0
            else:
                st_out = net(phi_frozen)
                s_out, t_out = st_out[:, 0:1], st_out[:, 1:2]

            s_out = asymmetric_soft_clamp(s_out, self.s_bounds[0], self.s_bounds[1])
            t_out = self.t_bounds[1] * torch.tanh(t_out / self.t_bounds[1])

            update_mask = 1.0 - current_mask
            phi = phi_frozen + update_mask * ((phi - t_out) * torch.exp(-s_out))
            log_det_jacobian += torch.sum(update_mask * (-s_out), dim=(1, 2, 3))
        return phi, log_det_jacobian


# ==========================================
# 6. 在线评估 (MCMC)
# ==========================================
def run_mcmc_evaluation(model, prior, config, total_n=10000, enforce_sym=False):
    model.eval()
    batch_size = config['batch_size']
    total_n = (total_n // batch_size) * batch_size
    dtype = torch.float64 if config.get('double_precision', False) else torch.float32

    all_phis = torch.empty((total_n, 1, config['L'], config['L']), dtype=dtype, device=device)
    all_s = torch.empty(total_n, dtype=dtype, device=device)
    all_log_qs = torch.empty(total_n, dtype=dtype, device=device)
    dummy_progress = torch.tensor(1.0, device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - i)
            z, log_p_z = prior.sample(current_batch)
            phi, log_det_J = model(z, dummy_progress, enforce_sym=enforce_sym)
            all_phis[i:i + current_batch] = phi
            all_s[i:i + current_batch] = compute_action(phi, config)
            all_log_qs[i:i + current_batch] = log_p_z - log_det_J

    s_np, log_q_np = all_s.cpu().numpy(), all_log_qs.cpu().numpy()
    log_rands_np = np.log(np.random.rand(total_n))
    accepted_indices = np.zeros(total_n, dtype=int)
    accepted_count = 0
    curr_s_val, curr_log_q_val = s_np[0], log_q_np[0]

    for i in range(1, total_n):
        prop_s_val, prop_log_q_val = s_np[i], log_q_np[i]
        log_acc_ratio = (-prop_s_val - prop_log_q_val) - (-curr_s_val - curr_log_q_val)
        if log_rands_np[i] < log_acc_ratio:
            curr_s_val, curr_log_q_val = prop_s_val, prop_log_q_val
            accepted_indices[i] = i
            accepted_count += 1
        else:
            accepted_indices[i] = accepted_indices[i - 1]

    idx_tensor = torch.tensor(accepted_indices, device=device, dtype=torch.long)
    chain_phis = all_phis[idx_tensor]
    phi_powers_mean, phi_powers_err = [], []
    for power in range(1, 6):
        pow_seq = (chain_phis ** power).mean(dim=(2, 3)).squeeze()
        phi_powers_mean.append(pow_seq.mean().item())
        phi_powers_err.append((pow_seq.std() / math.sqrt(total_n)).item())

    model.train()
    return accepted_count / (total_n - 1), phi_powers_mean, phi_powers_err


# ==========================================
# 7. 独立训练循环
# ==========================================
def train(config, resume=True):
    base_name = get_base_name(config)
    save_path = f"best_{base_name}.pt"
    checkpoint_path = f"latest_{base_name}.pt"
    observables_save_path = f"observables_{base_name}.npz"

    model = FlowModel(config).to(device)
    prior = StandardGaussianPrior(L=config['L']).to(device)

    if config['double_precision']: model, prior = model.double(), prior.double()

    eps_val = 1e-15 if config.get('double_precision', False) else 1e-8
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], eps=eps_val)

    history_loss, history_phi1, history_phi3, history_phi5 = [], [], [], []
    best_loss = float('inf')
    ema_loss = None
    start_iteration = 1
    achieved_milestones = set()

    if resume:
        old_checkpoint_path = auto_find_latest_checkpoint(config)
        if old_checkpoint_path and os.path.exists(old_checkpoint_path):
            print(f"🤖 正在从 {old_checkpoint_path} 恢复数据...")
            checkpoint = torch.load(old_checkpoint_path, map_location=device)
            clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(clean_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for param_group in optimizer.param_groups:
                param_group['initial_lr'], param_group['lr'] = config['lr'], config['lr']

            start_iteration = checkpoint['iteration'] + 1
            history_loss = checkpoint.get('history_loss', [])
            history_phi1 = checkpoint.get('history_phi1', [])
            history_phi3 = checkpoint.get('history_phi3', [])
            history_phi5 = checkpoint.get('history_phi5', [])
            ema_loss = checkpoint.get('ema_loss', None)
            best_loss = checkpoint.get('best_loss', min(history_loss) if history_loss else float('inf'))
            achieved_milestones = checkpoint.get('achieved_milestones', set())

    mcmc_steps, history_acc, history_phi_means, history_phi_errs = [], [], [], []
    if os.path.exists(observables_save_path):
        try:
            npz_data = np.load(observables_save_path)
            valid_idx = npz_data['steps'] < start_iteration
            mcmc_steps = npz_data['steps'][valid_idx].tolist()
            history_acc = npz_data['acc'][valid_idx].tolist()
            history_phi_means = npz_data['phi_means'][valid_idx].tolist()
            history_phi_errs = npz_data['phi_errs'][valid_idx].tolist()
        except Exception:
            pass

    scheduler = None
    if config.get('use_scheduler', True):
        safe_last_epoch = min(start_iteration - 1, config['scheduler_steps']) if start_iteration > 1 else -1
        if start_iteration > 1:
            eta_max, eta_min, T_max = config['lr'], config['scheduler_min'], config['scheduler_steps']
            exact_lr = eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * safe_last_epoch / T_max))
            for param_group in optimizer.param_groups: param_group['lr'] = exact_lr

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler_steps'],
                                                               eta_min=config['scheduler_min'],
                                                               last_epoch=safe_last_epoch)

    compiled_model = torch.compile(model) if hasattr(torch, 'compile') and device.type == 'cuda' else model
    model.train()

    for iteration in range(start_iteration, config['iterations'] + 1):
        optimizer.zero_grad()
        z, log_p_z = prior.sample(config['batch_size'])

        progress_val = min(iteration / config['warmup_steps'], 1.0)
        model.step_warmup(progress_val)
        progress_tensor = torch.tensor(progress_val, device=device,
                                       dtype=torch.float64 if config.get('double_precision', False) else torch.float32)

        enforce_sym = config.get('enforce_z2_sym', False) and (iteration >= config.get('sym_start_iter', 0))
        phi, log_det_J = compiled_model(z, progress_tensor, enforce_sym=enforce_sym)

        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi, config))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            optimizer.zero_grad()
            continue
        optimizer.step()

        if scheduler and iteration <= config['scheduler_steps']: scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        if iteration % 100 == 0:
            print(
                f"Iter: {iteration:6d} | EMA Loss: {ema_loss:.4f} | Best: {best_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': loss_val,
                        'best_loss': best_loss,
                        'history_loss': history_loss, 'ema_loss': ema_loss, 'achieved_milestones': achieved_milestones},
                       checkpoint_path)

        if ema_loss < best_loss and iteration >= 5000:
            best_loss = ema_loss
            torch.save({'model_state_dict': model.state_dict()}, save_path)

        if iteration >= 2000 and iteration % 2000 == 0:
            total_n = 20000 if iteration % 4000 != 0 else 50000
            print(f"\n🚀 [{config['exp_name']} - Iter {iteration}] MCMC 验证 (N={total_n})...")
            acc_rate, phi_means, phi_errs = run_mcmc_evaluation(compiled_model, prior, config, total_n=total_n,
                                                                enforce_sym=enforce_sym)

            print(f"📊 接受率: {acc_rate:.2%} | phi^1: {phi_means[0]:.6f}\n")
            mcmc_steps.append(iteration)
            history_acc.append(acc_rate)
            history_phi_means.append(phi_means)
            history_phi_errs.append(phi_errs)
            np.savez(observables_save_path, steps=np.array(mcmc_steps), acc=np.array(history_acc),
                     phi_means=np.array(history_phi_means), phi_errs=np.array(history_phi_errs))

    return model


# ==========================================
# 8. 自动化批处理执行入口
# ==========================================
if __name__ == "__main__":
    # 通用网络配置
    BASE_CONFIG = {
        'L': 14,
        'batch_size': 1024,
        'lr': 1e-3,
        'use_scheduler': True,
        'scheduler_min': 1e-5,
        'iterations': 15000,
        'scheduler_steps': 10000,
        'warmup_steps': 5000.0,
        'cnn_coupling_layers': 10,
        'branch_depth': 2,
        'double_precision': False,
        'trunk_channels': [16] * 12,
        'trunk_layers': [3] * 12,
        'trunk_use_multi_kernel': True,
        'trunk_kernel_sizes': (3,),
        'trunk_dilations': (1,),
        's_head_channels': [16] * 12,
        's_head_layers': [1] * 12,
        's_head_kernel_size': 3,
        't_head_channels': [16] * 12,
        't_head_layers': [1] * 12,
        't_kernel_size': 3,
    }

    if BASE_CONFIG.get('double_precision', False):
        torch.set_default_dtype(torch.float64)

    # ==================================
    # 实验 A 配置：纯动能项 (空间关联测试)
    # ==================================
    CONFIG_A = BASE_CONFIG.copy()
    CONFIG_A.update({
        'exp_name': 'Exp_A_Pure_Kinetic',
        'use_kinetic': True,
        'use_potential': False,
        'm_sq': 0.09,  # 必须正质量防止发散
        'lam': 0.0,
    })

    # ==================================
    # 实验 B 配置：纯势能项 (超局域非线性测试)
    # ==================================
    CONFIG_B = BASE_CONFIG.copy()
    CONFIG_B.update({
        'exp_name': 'Exp_B_Pure_Potential',
        'use_kinetic': False,
        'use_potential': True,
        'm_sq': -4.0,  # 必须正质量防止简并模式坍缩
        'lam': 5.113,
    })

    print("=" * 70)
    print("🔬 开始进行消融对照实验")
    print("=" * 70)

    print("\n>>> 正在启动 实验A：纯动能项 (测试空间关联学习瓶颈) <<<")
    train(config=CONFIG_A, resume=True)

    print("\n" + "=" * 70)

    print("\n>>> 正在启动 实验B：纯势能项 (测试局部非线性学习能力) <<<")
    train(config=CONFIG_B, resume=True)

    print("\n✅ 所有对照实验执行完毕！你可以利用保存的 npz 文件进行接受率曲线绘制了。")