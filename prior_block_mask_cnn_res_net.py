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

# ==========================================
# 0. 环境与设备配置
# ==========================================
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


def fmt_cfg(val):
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


# 找到文件中的 get_base_name 函数并替换为：
def get_base_name(config):
    """动态生成当前配置的专属文件名，包含 block_num 标识"""
    tk_str = '_'.join(map(str, config.get('trunk_kernel_sizes', (3, 3))))
    tdil_str = '_'.join(map(str, config.get('trunk_dilations', (1, 2))))
    trunk_rf = f"trk_{tk_str}_dil_{tdil_str}"

    # 🌟 修改为 prior_block 命名风格
    base_name = (
        f"prior_block_{config['block_num']}_dp_{config.get('double_precision', False)}_"
        f"L{config['L']}_c{config['cnn_coupling_layers']}_d{config.get('branch_depth', 2)}_"
        f"TrCh{fmt_cfg(config.get('trunk_channels'))}_Ly{fmt_cfg(config.get('trunk_layers'))}_{trunk_rf}_"
        f"Sh{fmt_cfg(config.get('s_head_channels'))}L{fmt_cfg(config.get('s_head_layers'))}_"
        f"Th{fmt_cfg(config.get('t_head_channels'))}L{fmt_cfg(config.get('t_head_layers'))}_"
        f"iter_{config['iterations']}"
    )
    return base_name


def auto_find_latest_checkpoint(config):
    base_name = get_base_name(config)
    # 通配符匹配已迭代的步数
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
    for file in get_sorted_files(""):
        filename = os.path.basename(file)
        if not filename.startswith(("latest_", "best_", "iter_")) and is_valid_checkpoint(file):
            return file
    return None


# ==========================================
# 2. 标量场理论的 Action 计算 (对齐 Z_2 提速版)
# ==========================================
laplacian_kernel = torch.tensor([[
    [0.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 0.0]
]], device=device).unsqueeze(1)


def compute_action(phi, config):
    phi_padded = F.pad(phi, pad=(1, 1, 1, 1), mode='circular')
    adaptive_kernel = laplacian_kernel.to(dtype=phi.dtype, device=phi.device)
    laplacian = F.conv2d(phi_padded, adaptive_kernel)
    action_density = phi * laplacian + config['m_sq'] * (phi ** 2) + config['lam'] * (phi ** 4)
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 动态掩码生成与卷积上下文网络 (剔除 Z_2 对称性出口)
# ==========================================
def create_dynamic_block_mask(L, block_num):
    """保留并适配了先前的 block 分割掩码逻辑"""
    if L % block_num != 0:
        raise ValueError(f"晶格大小 L ({L}) 必须能被 block_num ({block_num}) 整除！")
    block_size = L // block_num
    indices = torch.arange(L)
    block_indices = indices // block_size
    mask_2d = (block_indices[:, None] + block_indices[None, :]) % 2 == 0
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
        nn.init.zeros_(self.block[-2].weight)
        if self.block[-2].bias is not None:
            nn.init.zeros_(self.block[-2].bias)

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
                out = act_layer(conv_layer(out))
            outs.append(out)

        fused = torch.cat(outs, dim=1)
        out = self.fusion_conv(fused)
        return x + out


class ConvContextNet(nn.Module):
    def __init__(self, trunk_channels=64, trunk_layers=3, trunk_use_multi_kernel=True,
                 trunk_kernel_sizes=(3, 3), trunk_dilations=(1, 2),
                 s_head_channels=32, s_head_layers=1, s_head_kernel_size=3,
                 t_head_channels=32, t_head_layers=1, t_kernel_size=3, branch_depth=2):
        super().__init__()
        self.s_head_layers = s_head_layers
        self.t_head_layers = t_head_layers

        # 1. 构造共享主干网络
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

        # 2. 构造 S 的独立分支
        s_pad = s_head_kernel_size // 2
        if s_head_layers == 0:
            self.s_head = nn.Conv2d(trunk_channels, 1, kernel_size=s_head_kernel_size, padding=s_pad,
                                    padding_mode='circular')
        else:
            s_head_list = [nn.Conv2d(trunk_channels, s_head_channels, kernel_size=s_head_kernel_size, padding=s_pad,
                                     padding_mode='circular'), nn.LeakyReLU(0.01)]
            for _ in range(s_head_layers): s_head_list.append(
                ResBlock(s_head_channels, kernel_size=s_head_kernel_size, branch_depth=branch_depth))
            s_head_list.append(nn.Conv2d(s_head_channels, 1, kernel_size=1))
            self.s_head = nn.Sequential(*s_head_list)

        # 3. 构造 T 的独立分支
        t_pad = t_kernel_size // 2
        if t_head_layers == 0:
            self.t_head = nn.Conv2d(trunk_channels, 1, kernel_size=t_kernel_size, padding=t_pad,
                                    padding_mode='circular')
        else:
            t_head_list = [nn.Conv2d(trunk_channels, t_head_channels, kernel_size=t_kernel_size, padding=t_pad,
                                     padding_mode='circular'), nn.LeakyReLU(0.01)]
            for _ in range(t_head_layers): t_head_list.append(
                ResBlock(t_head_channels, kernel_size=t_kernel_size, branch_depth=branch_depth))
            t_head_list.append(nn.Conv2d(t_head_channels, 1, kernel_size=1))
            self.t_head = nn.Sequential(*t_head_list)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.trunk_net[0].weight, mean=0, std=0.1)
        for head, layers in [(self.s_head, self.s_head_layers), (self.t_head, self.t_head_layers)]:
            if layers == 0:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            else:
                nn.init.normal_(head[0].weight, mean=0, std=0.05)
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        shared_feat = self.trunk_net(x)
        s_out = self.s_head(shared_feat)
        t_out = self.t_head(shared_feat)
        return torch.cat([s_out, t_out], dim=1)


# ==========================================
# 4. 自由场先验 (Free Field Prior)
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
# 5. 流模型 (结合了 Block Mask 和 防爆盾)
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

        # 🌟 挂载动态 Block 掩码
        self.register_buffer('base_mask', create_dynamic_block_mask(self.L, config['block_num']))

        self.context_nets = nn.ModuleList()
        dtype = torch.float64 if config.get('double_precision', False) else torch.float32

        self.register_buffer('s_bounds', torch.tensor([-0.2, 3.0], dtype=dtype))
        self.register_buffer('t_bounds', torch.tensor([-3.0, 3.0], dtype=dtype))

        def get_layer_cfg(key, default, layer_idx):
            val = config.get(key, default)
            if isinstance(val, (list, tuple)):
                assert len(val) == self.cnn_layers, f"❌ 列表参数不匹配"
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

    def step_warmup(self, progress):
        clamped_progress = min(progress, 1.0)
        self.s_bounds[0].fill_(-0.2 - 0.3 * clamped_progress)
        self.s_bounds[1].fill_(3.0 + 2.0 * clamped_progress)
        self.t_bounds[0].fill_(-3.0 - 5.0 * clamped_progress)
        self.t_bounds[1].fill_(3.0 + 5.0 * clamped_progress)

    def forward(self, z, progress=None, enforce_sym=False):
        phi = z
        log_det_jacobian = 0

        for step, net in enumerate(self.context_nets):
            current_mask = self.base_mask if step % 2 == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi

            if enforce_sym:
                st_out_pos = net(phi_frozen)
                st_out_neg = net(-phi_frozen)
                s_out = (st_out_pos[:, 0:1, :, :] + st_out_neg[:, 0:1, :, :]) / 2.0
                t_out = (st_out_pos[:, 1:2, :, :] - st_out_neg[:, 1:2, :, :]) / 2.0
            else:
                st_out = net(phi_frozen)
                s_out, t_out = st_out[:, 0:1, :, :], st_out[:, 1:2, :, :]

            s_out = asymmetric_soft_clamp(s_out, self.s_bounds[0], self.s_bounds[1])
            t_out = self.t_bounds[1] * torch.tanh(t_out / self.t_bounds[1])

            update_mask = 1.0 - current_mask
            phi = phi_frozen + update_mask * ((phi - t_out) * torch.exp(-s_out))
            log_det_jacobian += torch.sum(update_mask * (-s_out), dim=(1, 2, 3))

        return phi, log_det_jacobian


# ==========================================
# 6. 在线评估 (MCMC CPU 优化版)
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

    s_np = all_s.cpu().numpy()
    log_q_np = all_log_qs.cpu().numpy()
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
    # 🌟 挂载纯净的标准高斯采样先验
    prior = FreeFieldPrior(L=config['L'], m_sq_prior=4.0).to(device)

    if config['double_precision']:
        model, prior = model.double(), prior.double()

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

    if hasattr(torch, 'compile') and device.type == 'cuda':
        compiled_model = torch.compile(model)
    else:
        compiled_model = model

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

        with torch.no_grad():
            history_phi1.append(torch.mean(phi).item())
            history_phi3.append(torch.mean(phi ** 3).item())
            history_phi5.append(torch.mean(phi ** 5).item())

        V = config['L'] ** 2
        loss_sym = V * (torch.mean(phi) ** 2)

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
                f"Iter: {iteration:6d} | "
                f"Z2: {'ON' if enforce_sym else 'OFF'} "
                f"| 瞬时 Loss: {loss_val:.4f} "
                f"| EMA Loss: {ema_loss:.4f} | Best: {best_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': loss_val,
                        'best_loss': best_loss,
                        'history_loss': history_loss, 'ema_loss': ema_loss, 'achieved_milestones': achieved_milestones},
                       checkpoint_path)

        if ema_loss < best_loss and iteration >= 10000:
            best_loss = ema_loss
            torch.save({'model_state_dict': model.state_dict()}, save_path)

        if iteration >= 4000 and iteration % 2000 == 0:
            total_n = 50000 if iteration % 5000 != 0 else 100000
            print(f"\n🚀 [Iter {iteration}] MCMC 验证 (N={total_n})...")
            acc_rate, phi_means, phi_errs = run_mcmc_evaluation(compiled_model, prior, config, total_n=total_n,
                                                                enforce_sym=enforce_sym)

            print(f"📊 接受率: {acc_rate:.2%} | phi^1: {phi_means[0]:.6f}\n")
            mcmc_steps.append(iteration)
            history_acc.append(acc_rate)
            history_phi_means.append(phi_means)
            history_phi_errs.append(phi_errs)
            np.savez(observables_save_path, steps=np.array(mcmc_steps), acc=np.array(history_acc),
                     phi_means=np.array(history_phi_means), phi_errs=np.array(history_phi_errs))

            if acc_rate >= config['target_acc_ratio']:
                print(f"🎉 提前达标！({acc_rate:.2%} >= {config['target_acc_ratio']:.0%})")
                break
    return model


# ==========================================
# 8. 自动化批处理执行入口
# ==========================================
if __name__ == "__main__":
    # 配置使用全新的 U-Net Sandglass 样式
    CONFIG = {
        'type': 'gaussian_prior_block_mask',
        'L': 8,
        'm_sq': -4.0,
        'lam': 6.008,
        'batch_size': 1024,
        'lr': 1e-3,
        'use_scheduler': True,
        'scheduler_min': 1e-5,
        'iterations': 20000,
        'scheduler_steps': 20000,
        'warmup_steps': 5000.0,
        'target_acc_ratio': 0.78,

        'enforce_z2_sym': False,
        'sym_start_iter': 30000,

        'cnn_coupling_layers': 10,
        'branch_depth': 2,
        'double_precision': False,

        # 🌟 统一配置通道数量
        'trunk_channels': [16] * 10,
        'trunk_layers': [3] * 10,
        'trunk_use_multi_kernel': True,
        'trunk_kernel_sizes': (3,),
        'trunk_dilations': (1,),

        's_head_channels': [16] * 10,
        's_head_layers': [1] * 10,
        's_head_kernel_size': 3,

        't_head_channels': [16] * 10,
        't_head_layers': [1] * 10,
        't_kernel_size': 3,
    }

    if CONFIG.get('double_precision', False):
        torch.set_default_dtype(torch.float64)

    # 🌟 触发原始的 Block Num 测试逻辑
    block_nums_to_test = [2, 4, 8]

    for b_num in block_nums_to_test:
        print("=" * 60)
        print(f"🚀 开始执行 Block Mask 实验: block_num = {b_num} (晶格 L={CONFIG['L']})")
        print("=" * 60)

        current_config = CONFIG.copy()
        current_config['block_num'] = b_num

        train(config=current_config, resume=True)