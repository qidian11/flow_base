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
        f"z_2_symmetry_cnn_res_independent_coupling_model_double_precision_*_"
        f"{config['L']}_coupling_layers_{(config['cnn_coupling_layers'])}_"
        f"{k_str}_"
        f"hidden_layers_{config['hidden_layers']}_hidden_channels_{config['hidden_channels']}_"
        f"iterations_*.pt"
    )

    def get_sorted_files(prefix):
        files = glob.glob(prefix + base_pattern)
        return sorted(files, key=os.path.getmtime, reverse=True)

    for file in get_sorted_files("latest_"):
        if is_valid_checkpoint(file):
            return file
        else:
            print(f"⚠️ 警告: 检测到损坏的 latest 断点并已自动跳过 -> {file}")

    for file in get_sorted_files("best_"):
        if is_valid_checkpoint(file):
            print(f"🔄 未找到可用的 latest，已自动回退到最新的 best 断点 -> {file}")
            return file
        else:
            print(f"⚠️ 警告: 检测到损坏的 best 断点并已自动跳过 -> {file}")

    for file in get_sorted_files(""):
        filename = os.path.basename(file)
        if not filename.startswith(("latest_", "best_")):
            if is_valid_checkpoint(file):
                print(f"🔄 best 也不可用，已自动回退到里程碑断点 -> {file}")
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
    'scheduler_min': 1e-5,
    'iterations': 20000,

    'cnn_coupling_layers': 12,
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

save_path = f"best_z_2_symmetry_cnn_res_independent_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
loss_save_path = f"z_2_symmetry_cnn_res_independent_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npy"
checkpoint_path = f"latest_z_2_symmetry_cnn_res_independent_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
phi_ensemble_save_path = f"phi_ensemble_z_2_symmetry_cnn_res_independent_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_loss_history_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npz"

CONFIG['save_path'] = save_path
CONFIG['loss_save_path'] = loss_save_path
CONFIG['checkpoint_path'] = checkpoint_path
CONFIG['phi_ensemble_save_path'] = phi_ensemble_save_path


# ==========================================
# 2. 标量场理论的 Action 计算
# ==========================================
def compute_action(phi):
    phi_up = torch.roll(phi, shifts=-1, dims=2)
    phi_down = torch.roll(phi, shifts=1, dims=2)
    phi_left = torch.roll(phi, shifts=-1, dims=3)
    phi_right = torch.roll(phi, shifts=1, dims=3)
    laplacian = 4 * phi - phi_up - phi_down - phi_left - phi_right
    action_density = phi * laplacian + CONFIG['m_sq'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 掩码生成与卷积上下文网络
# ==========================================
def create_checkerboard_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()

# =========================================
# 使用奇函数作为激活函数
class LeakyTanh(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        # 预先计算好常数，避免在前向传播中重复计算 1 - alpha
        self.alpha = alpha
        self.beta = 1.0 - alpha

    def forward(self, x):
        # 极简且高效的纯前向计算
        return self.alpha * x + self.beta * torch.tanh(x)
# ==========================================


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding,
                               padding_mode='circular', bias=False)
        self.act1 = LeakyTanh()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding,
                               padding_mode='circular', bias=False)
        self.act2 = LeakyTanh()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding,
                               padding_mode='circular', bias=False)
        self.act3 = LeakyTanh()

        # 🌟 模块自治：自己管好自己的初始化
        # 🌟 改进：将残差块的最后一层严格置零
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        return x + self.act3(self.conv3(self.act2(self.conv2(self.act1(self.conv1(x))))))


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1), branch_depth=3):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "卷积核数量和空洞率数量必须严格匹配！"

        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            assert k % 2 != 0, f"多尺度卷积核必须均为奇数！当前输入了偶数核: {k}"
            pad = d * (k - 1) // 2
            layers = []
            for _ in range(branch_depth):
                layers.append(
                    nn.Conv2d(channels, channels, kernel_size=k, stride=1, padding=pad, dilation=d,
                              padding_mode='circular', bias=False)
                )
                layers.append(LeakyTanh())
            self.branches.append(nn.Sequential(*layers))

        num_branches = len(kernel_sizes)
        self.fusion_conv = nn.Conv2d(channels * num_branches, channels, kernel_size=1, bias=False)
        # 🌟 模块自治：分支卷积给 0.01，出口融合卷积给 0
        for branch in self.branches:
            for m in branch:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(self.fusion_conv.weight)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.fusion_conv(out)
        return x + out


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4, kernel_size=3, use_multi_kernel=False,
                 multi_kernel_sizes=(3, 5, 7), multi_kernel_dilations=(1, 1, 1), branch_depth=3):
        super().__init__()

        # 对s通道的输出再卷积一次conv(h_s**2)
        self.s_exit_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='circular', bias=True)
        # 将出口卷积的权重和偏置严格初始化为 0
        nn.init.zeros_(self.s_exit_conv.weight)
        nn.init.zeros_(self.s_exit_conv.bias)

        layers = []

        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=1, stride=1,
                                padding=0, padding_mode='circular', bias=False)
                      )
        layers.append(LeakyTanh())

        for _ in range(num_hidden_layers):
            if use_multi_kernel:
                layers.append(MultiScaleResBlock(hidden_channels, kernel_sizes=multi_kernel_sizes,
                                                 dilations=multi_kernel_dilations, branch_depth=branch_depth))
            else:
                layers.append(ResBlock(hidden_channels, kernel_size=kernel_size))

        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=1,
                                stride=1, padding=0, padding_mode='circular',bias=False)
                      )
        self.net = nn.Sequential(*layers)

        # 头部给予 0.1 小权重防激活饱和
        nn.init.normal_(self.net[0].weight, mean=0, std=0.1)

        # 1. 尾部负责 h_s (通道 0) 的部分：必须给健康的噪声，保证 h_s**2 不为 0！
        nn.init.normal_(self.net[-1].weight[0:1], mean=0, std=0.1)

        # 2. 尾部负责 h_t (通道 1) 的部分：绝对为 0，保证初始恒等映射和奇函数对称性
        nn.init.zeros_(self.net[-1].weight[1:2])

        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x)
        h_s = out[:, 0:1, :, :]
        h_t = out[:, 1:2, :, :]
        h_s= self.s_exit_conv(h_s**2)
        out = torch.cat([h_s, h_t], dim=1)
        return out


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
        self.s_biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.total_layers)])

        for _ in range(self.cnn_layers):
            self.context_nets.append(
                ConvContextNet(
                    hidden_channels=config['hidden_channels'],
                    num_hidden_layers=config['hidden_layers'],
                    kernel_size=config['kernel_size'],
                    use_multi_kernel=config.get('use_multi_kernel', False),
                    multi_kernel_sizes=config.get('multi_kernel_sizes', (3, 3)),
                    multi_kernel_dilations=config.get('multi_kernel_dilations', (1, 2)),
                    branch_depth=config.get('branch_depth', 3)
                )
            )

        if config.get('use_multi_kernel', False):
            arch_info = f"多尺度: {config.get('multi_kernel_sizes')} | 空洞率: {config.get('multi_kernel_dilations')}"
        else:
            arch_info = f"单核: {config['kernel_size']}"

        print("=" * 70)
        print(f"🌟 物理流模型 (UV-IR Decoupling 架构) 初始化完毕！")
        print(f"👉 总耦合层数: {self.total_layers} 层 (特征通道数: {config['hidden_channels']})")
        print(f"   │")
        print(f"   ├─ [UV 物理] 纯 CNN 局域重整化: 前 {self.cnn_layers} 层")
        print(f"      ├─ 内部隐藏层 (ResBlocks): {config['hidden_layers']} 层 / 耦合层")
        print(f"      └─ 卷积网络配置: {arch_info} | 分支深度: {config.get('branch_depth', 3)}")
        print("=" * 70)

    def forward(self, z, s_clamp_min=None, s_clamp_max=None, t_clamp_min=None, t_clamp_max=None):
        phi = z
        log_det_jacobian = 0

        for i, net in enumerate(self.context_nets):
            current_mask = self.base_mask if (i % 2) == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi

            st_out = net(phi_frozen)
            # 🌟 提取两个通道，s_out是严格的偶函数，t_out是严格的奇函数！
            s_out = st_out[:, 0:1, :, :]
            t_out = st_out[:, 1:2, :, :]

            if self.training:
                s_out = asymmetric_soft_clamp(s_out, min_val=s_clamp_min, max_val=s_clamp_max)
                t_out = clamp_t(t_out, min_val=t_clamp_min, max_val=t_clamp_max)

            update_mask = 1.0 - current_mask
            # 🌟 更新 1: 严格遵循论文 Eq. 10 的生成映射 (Prior -> Data)
            # 公式: phi_b = (z_b - t) ⊙ exp(-s)
            phi = phi_frozen + update_mask * ((phi - t_out) * torch.exp(-s_out))

            # 🌟 更新 2: 修正雅可比行列式的对数
            # 由于 ∂phi / ∂z = exp(-s)，因此 log_det 是 -s
            log_det_jacobian += torch.sum(update_mask * (-s_out), dim=(1, 2, 3))

        return phi, log_det_jacobian


def asymmetric_soft_clamp(x, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        return x

    pos_scale = max_val
    neg_scale = abs(min_val)

    pos_val = pos_scale * torch.tanh(x / pos_scale)
    neg_val = neg_scale * torch.tanh(x / neg_scale)

    return torch.where(x >= 0, pos_val, neg_val)


def clamp_t(x, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        return x
    return torch.clamp(x, min_val, max_val)


# ==========================================
# 6. 自训练循环
# ==========================================
def train(save_path=CONFIG['save_path'], loss_save_path=CONFIG['loss_save_path'], resume=True,
          checkpoint_path=CONFIG['checkpoint_path']):
    model = FlowModel(CONFIG).to(device)
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=abs(CONFIG['m_sq'])).to(device)

    if CONFIG['double_precision']:
        model = model.double()
        prior = prior.double()

    if hasattr(torch, 'compile') and device.type == 'cuda':
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = None
    if CONFIG.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['iterations'],
                                                               eta_min=CONFIG['scheduler_min'])

    history_loss = []
    best_loss = float('inf')
    ema_loss = None
    start_iteration = 1

    if resume:
        old_checkpoint_path = auto_find_latest_checkpoint(CONFIG)
        if old_checkpoint_path and os.path.exists(old_checkpoint_path):
            print(f"检测到断点文件，正在从 {old_checkpoint_path} 恢复训练...")
            checkpoint = torch.load(old_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if CONFIG.get('double_precision', False):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.double()

            if CONFIG.get('use_scheduler', True):
                if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint[
                    'scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = CONFIG['lr']

            start_iteration = checkpoint['iteration'] + 1
            if 'history_loss' in checkpoint: history_loss = checkpoint['history_loss']

            if 'ema_loss' in checkpoint:
                ema_loss = checkpoint['ema_loss']

            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
            elif len(history_loss) > 0:
                best_loss = min(history_loss)

            print(f"恢复成功！将从第 {start_iteration} 步继续训练。")
        else:
            print("未找到断点文件，将从头开始训练。")

    model.train()
    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()
        z, log_p_z = prior.sample(CONFIG['batch_size'])

        # 🌟 更新 3: 配合 exp(-s) 翻转 Clamp 限制区间
        if iteration < 3000:
            s_c_min, s_c_max = -0.5, 4.0  # 以前是 -4.0, 0.5
            t_c_min, t_c_max = -15.0, 15.0
        elif iteration < 8000:
            s_c_min, s_c_max = -0.8, 5.0  # 以前是 -5.0, 0.8
            t_c_min, t_c_max = -15.0, 15.0
        else:
            s_c_min, s_c_max = None, None
            t_c_min, t_c_max = None, None

        phi, log_det_J = model(z, s_c_min, s_c_max, t_c_min, t_c_max)

        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None: scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)

        # 🌟 修复：计算 Loss 的指数移动平均 (EMA)，滤除蒙特卡洛采样噪声
        if ema_loss is None:
            ema_loss = loss_val
        else:
            # 0.95 的动量意味着参考过去约 20 个 batch (10000+ 个样本) 的真实物理表现
            ema_loss = 0.95 * ema_loss + 0.05 * loss_val

        if iteration % 100 == 0:
            # 打印时同时显示瞬时 Loss 和真实的 EMA Loss
            print(
                f"迭代 {iteration:6d}/{CONFIG['iterations']} | 瞬时 Loss: {loss_val:.4f} | 平滑 Loss: {ema_loss:.4f} | Best: {best_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': loss_val,
                        'best_loss': best_loss, 'history_loss': history_loss,
                        'ema_loss': ema_loss,}, checkpoint_path)

        # 🌟 修复：用剔除噪声后的 ema_loss 去竞选 Best Model
        if ema_loss < best_loss and iteration >= 10000:
            best_loss = ema_loss
            print(f"🌟 第 {iteration}步发现更优全局模型！当前最佳平滑 Loss: {best_loss:.4f}，正在更新 best 断点...")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': loss_val,
                        'best_loss': best_loss, 'history_loss': history_loss,
                        'ema_loss': ema_loss,}, save_path)

        if iteration % 1000 == 0:
            np.save(loss_save_path, np.array(history_loss))
            milestone_path = f"z_2_symmetry_shared_coupling_model_double_precision_{CONFIG['double_precision']}_{CONFIG['L']}_coupling_layers_{total_coupling_layers}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{iteration}.pt"

            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': loss_val,
                        'best_loss': best_loss, 'history_loss': history_loss,
                        'ema_loss': ema_loss,}, milestone_path)
    print("训练结束！")


if __name__ == "__main__":
    if CONFIG.get('double_precision', False): torch.set_default_dtype(torch.float64)
    train()