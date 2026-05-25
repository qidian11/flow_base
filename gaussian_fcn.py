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
    # 🌟 修改点：针对 FCN 简化的存档匹配逻辑
    base_pattern = (
        f"gaussian_fcn_dp_{config['double_precision']}_"
        f"{config['L']}_coupling_layers_{config['fcn_coupling_layers']}_"
        f"hidden_layers_{config['hidden_layers']}_features_{config['hidden_features']}_"
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
# 1. 物理参数配置 (清理了所有 CNN 特有的参数)
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
    'warmup_steps': 5000.0,
    'target_acc_ratio': 0.78,

    # 🌟 修改点：FCN 架构参数
    'fcn_coupling_layers': 12,
    'hidden_layers': 8,  # MLP 的隐藏层数量
    'hidden_features': 448,  # MLP 的隐藏层神经元宽度
    'double_precision': False,
}

total_coupling_layers = CONFIG['fcn_coupling_layers']

# 🌟 修改点：更新存档文件命名，使用 fcn 标识
file_suffix = f"gaussian_fcn_dp_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_hidden_layers_{CONFIG['hidden_layers']}_features_{CONFIG['hidden_features']}_iterations_{CONFIG['iterations']}"

save_path = f"best_{file_suffix}.pt"
loss_save_path = f"loss_{file_suffix}.npy"
checkpoint_path = f"latest_{file_suffix}.pt"
phi_ensemble_save_path = f"phi_ensemble_{file_suffix}.npz"

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
# 2. 标量场理论的 Action 计算
# ==========================================
def compute_action(phi):
    phi_padded = F.pad(phi, pad=(1, 1, 1, 1), mode='circular')
    laplacian = F.conv2d(phi_padded, laplacian_kernel)
    action_density = phi * laplacian + CONFIG['m_sq'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 掩码生成与全连接上下文网络
# ==========================================
def create_checkerboard_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()


# 🌟 修改点：全新的 FCN 架构替换掉了 CNN
class FCNContextNet(nn.Module):
    def __init__(self, L, hidden_features=256, num_hidden_layers=4):
        super().__init__()
        self.L = L
        self.in_features = L * L

        def build_independent_mlp():
            layers = []
            layers.append(nn.Linear(self.in_features, hidden_features))
            layers.append(nn.LeakyReLU(0.01))

            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(hidden_features, hidden_features))
                layers.append(nn.LeakyReLU(0.01))

            layers.append(nn.Linear(hidden_features, self.in_features))
            return nn.Sequential(*layers)

        self.s_net = build_independent_mlp()
        self.t_net = build_independent_mlp()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.s_net[0].weight, mean=0, std=0.1)
        nn.init.zeros_(self.s_net[-1].weight)
        nn.init.zeros_(self.s_net[-1].bias)

        nn.init.normal_(self.t_net[0].weight, mean=0, std=0.1)
        nn.init.zeros_(self.t_net[-1].weight)
        nn.init.zeros_(self.t_net[-1].bias)

    def forward(self, x):
        # x shape: (B, 1, L, L)
        B = x.shape[0]
        # 展平输入以适应全连接层
        x_flat = x.view(B, -1)

        # 分别计算 s 和 t，并重新 reshape 回 2D 网格结构
        s_out = self.s_net(x_flat).view(B, 1, self.L, self.L)
        t_out = self.t_net(x_flat).view(B, 1, self.L, self.L)

        return torch.cat([s_out, t_out], dim=1)


# ==========================================
# 4. 标准高斯先验
# ==========================================
class StandardGaussianPrior(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.V = L * L

        self.register_buffer('const_factor', torch.tensor(0.5 * self.V * math.log(2.0 * math.pi)))

    def sample(self, batch_size):
        z = torch.randn(batch_size, 1, self.L, self.L,
                        device=self.const_factor.device,
                        dtype=self.const_factor.dtype)

        log_p_z = -0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - self.const_factor
        return z, log_p_z


# ==========================================
# 5. 流模型
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config['L']
        self.total_layers = config['fcn_coupling_layers']
        self.register_buffer('base_mask', create_checkerboard_mask(self.L))
        self.context_nets = nn.ModuleList()
        dtype = torch.float64 if config.get('double_precision', False) else torch.float32

        self.register_buffer('s_bounds', torch.tensor([-0.5, 4.0], dtype=dtype))
        self.register_buffer('t_bounds', torch.tensor([-15.0, 15.0], dtype=dtype))

        # 🌟 修改点：实例化 FCN 上下文网络
        for _ in range(self.total_layers):
            self.context_nets.append(FCNContextNet(
                L=config['L'],
                hidden_features=config['hidden_features'],
                num_hidden_layers=config['hidden_layers']
            ))

        print("=" * 70)
        print(f"🌟 Prior_FCN 物理流模型初始化完毕！(标准高斯先验)")
        print(f"👉 总耦合层数: {self.total_layers} 层 (双步复用, FCN 特征宽度: {config['hidden_features']})")
        print(f"   │")
        print(f"   ├─ 内部隐藏层 (MLP): {config['hidden_layers']} 层 / 耦合层")
        print(f"   └─ 全连接展平维度: {self.L * self.L} -> {config['hidden_features']} -> {self.L * self.L}")
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

    def forward(self, z, progress=None):
        phi = z
        log_det_jacobian = 0

        for net in self.context_nets:
            for step in range(2):
                current_mask = self.base_mask if step == 0 else (1.0 - self.base_mask)
                phi_frozen = current_mask * phi

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
# 7. 训练主循环
# ==========================================
def train():
    model = FlowModel(CONFIG).to(device)
    prior = StandardGaussianPrior(L=CONFIG['L']).to(device)

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