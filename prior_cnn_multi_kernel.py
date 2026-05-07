import torch
import torch.nn as nn
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
    # 根据新增加的开关，决定搜索哪种架构的文件
    if config.get('use_multi_kernel', False):
        sizes_str = '_'.join(map(str, config['multi_kernel_sizes']))
        dilations_str = '_'.join(map(str, config.get('multi_kernel_dilations', (1, 1, 1))))
        k_str = f"multi_kernels_{sizes_str}_dilations_{dilations_str}"
    else:
        k_str = f"kernel_size_{config['kernel_size']}"

    base_pattern = (
        f"prior_cnn_res_model_double_precision_*_"
        f"{config['L']}_coupling_layers_{config['coupling_layers']}_"
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
    # --- 原有字段 (严格保持不变) ---
    'L': 14,
    'm_sq': -4.0,
    'lam': 5.113,
    'batch_size': 512,
    'lr': 3e-6,
    'use_scheduler': False,
    'iterations': 40000,
    'coupling_layers': 14,
    'kernel_size': 14,  # 原汁原味的旧参数，如果你关掉多核开关，网络就会用这个
    'hidden_layers': 4,
    'hidden_channels': 128,
    'double precision': False,

    # === 新增字段 (控制多尺度卷积) ===
    'use_multi_kernel': True,  # 设为 False 则退回你的原版单架构网络
    'multi_kernel_sizes': (1, 3, 5),  # 这里自定义多尺度的大小
    'multi_kernel_dilations': (1, 1, 1), # <--- 在这里任意修改每个核的空洞率！
}

# 动态生成文件名中的标识（加入 dilations）
if CONFIG.get('use_multi_kernel', False):
    sizes_str = '_'.join(map(str, CONFIG['multi_kernel_sizes']))
    dilations_str = '_'.join(map(str, CONFIG.get('multi_kernel_dilations', (1, 1, 1))))
    k_str = f"multi_kernels_{sizes_str}_dilations_{dilations_str}"
else:
    k_str = f"kernel_size_{CONFIG['kernel_size']}"


# 文件名生成
save_path = f"best_prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
loss_save_path = f"prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_loss_history_coupling_layers_{CONFIG['coupling_layers']}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npy"
checkpoint_path = f"latest_prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
phi_ensemble_save_path = f"phi_ensemble_prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_loss_history_coupling_layers_{CONFIG['coupling_layers']}_{k_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npz"

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


# --- 你的原版单分支网络 (已修复 padding 越界 bug) ---
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding,
                               padding_mode='circular')
        self.act1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding,
                               padding_mode='circular')
        self.act2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding,
                               padding_mode='circular')
        self.act3 = nn.LeakyReLU(0.01)

    def forward(self, x):
        return x + self.act3(self.conv3(self.act2(self.conv2(self.act1(self.conv1(x))))))


# --- 新的多尺度网络 ---
class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 5, 7), dilations=(1, 1, 1)):
        super().__init__()
        k1, k2, k3 = kernel_sizes
        d1, d2, d3 = dilations
        assert k1 % 2 != 0 and k2 % 2 != 0 and k3 % 2 != 0, "多尺度卷积核必须均为奇数！"

        self.conv_a = nn.Conv2d(channels, channels, kernel_size=k1, stride=1, padding=k1 // 2, dilation=d1, padding_mode='circular')
        self.conv_b = nn.Conv2d(channels, channels, kernel_size=k2, stride=1, padding=k2 // 2, dilation=d2,padding_mode='circular')
        self.conv_c = nn.Conv2d(channels, channels, kernel_size=k3, stride=1, padding=k3 // 2, dilation=d3,padding_mode='circular')

        self.act = nn.LeakyReLU(0.01)
        self.fusion_conv = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x):
        out_a = self.conv_a(x)
        out_b = self.conv_b(x)
        out_c = self.conv_c(x)
        out = torch.cat([out_a, out_b, out_c], dim=1)
        out = self.act(out)
        out = self.fusion_conv(out)
        return x + out


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4, kernel_size=3, use_multi_kernel=False,
                 multi_kernel_sizes=(3, 5, 7), multi_kernel_dilations=(1, 1, 1)):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=1, stride=1, padding=0, padding_mode='circular'))
        layers.append(nn.LeakyReLU(0.01))

        # 🌟 智能路由：根据新增的开关决定使用哪种块
        for _ in range(num_hidden_layers):
            if use_multi_kernel:
                layers.append(MultiScaleResBlock(hidden_channels, kernel_sizes=multi_kernel_sizes, dilations=multi_kernel_dilations))
            else:
                layers.append(ResBlock(hidden_channels, kernel_size=kernel_size))

        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=1, stride=1, padding=0, padding_mode='circular'))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


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
        self.coupling_layers = config['coupling_layers']
        self.register_buffer('base_mask', create_checkerboard_mask(self.L))

        # 将参数从 config 解包并向下透传
        self.context_nets = nn.ModuleList([
            ConvContextNet(
                hidden_channels=config['hidden_channels'],
                num_hidden_layers=config['hidden_layers'],
                kernel_size=config['kernel_size'],
                use_multi_kernel=config.get('use_multi_kernel', False),
                multi_kernel_sizes=config.get('multi_kernel_sizes', (3, 5, 7)),
                multi_kernel_dilations=config.get('multi_0kernel_dilations', (1, 1, 1)),
            ) for _ in range(self.coupling_layers)
        ])

        arch_info = f"多尺度组合: {config.get('multi_kernel_sizes')}" if config.get(
            'use_multi_kernel') else f"单核: {config['kernel_size']}"
        print(
            f"当前模型耦合层：{self.coupling_layers} 通道数：{config['hidden_channels']}， 隐藏层：{config['hidden_layers']}, 卷积配置：{arch_info}")

    def forward(self, z, iteration=None):
        phi = z
        log_det_jacobian = 0
        for i in range(self.coupling_layers):
            current_mask = self.base_mask if i % 2 == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi
            st_out = self.context_nets[i](phi_frozen)
            s_out = st_out[:, 0:1, :, :]
            t_out = st_out[:, 1:2, :, :]

            # ================= 🌟 智能状态感知退火逻辑 =================
            if self.training:
                # 【训练阶段】：需要保护措施
                if iteration is not None:
                    if iteration < 5000:
                        s_out = torch.clamp(s_out, min=-6.0, max=3.0)
                    elif iteration < 10000:
                        s_out = torch.clamp(s_out, min=-10.0, max=5.0)
                    # iteration >= 15000 视为完全放开
                else:
                    # 万一你外部漏传了 iteration，给个保底兜底
                    s_out = torch.clamp(s_out, min=-8.0, max=4.0)
            else:
                # 【生成系综阶段】：model.eval() 被调用，彻底解除封印！
                # 不做任何 clamp 操作，让模型完全自由发挥它学到的真实物理分布
                pass
                # =========================================================

            update_mask = 1.0 - current_mask
            phi = phi_frozen + update_mask * (phi * torch.exp(s_out) + t_out)
            log_det_jacobian += torch.sum(update_mask * s_out, dim=(1, 2, 3))
        return phi, log_det_jacobian


# ==========================================
# 6. 自训练循环
# ==========================================
def train(save_path=CONFIG['save_path'], loss_save_path=CONFIG['loss_save_path'], resume=True,
          checkpoint_path=CONFIG['checkpoint_path']):
    # 实例化流模型 (传入整个 CONFIG 方便管理)
    model = FlowModel(CONFIG).to(device)

    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=abs(CONFIG['m_sq'])).to(device)

    if CONFIG['double precision']:
        model = model.double()
        prior = prior.double()

    if hasattr(torch, 'compile') and device.type == 'cuda':
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = None
    if CONFIG.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['iterations'], eta_min=1e-5)

    history_loss = []
    best_loss = float('inf')
    start_iteration = 1

    if resume:
        old_checkpoint_path = auto_find_latest_checkpoint(CONFIG)
        if old_checkpoint_path and os.path.exists(old_checkpoint_path):
            print(f"检测到断点文件，正在从 {old_checkpoint_path} 恢复训练...")
            checkpoint = torch.load(old_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if CONFIG.get('double precision', False):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.double()

            # ================= 🌟 智能学习率与调度器恢复逻辑 =================
            if CONFIG.get('use_scheduler', True):
                # 【模式 A】启用 Scheduler：恢复 Scheduler 的记忆，由它全权接管学习率
                if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint[
                    'scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                # 【模式 B】关闭 Scheduler：强制用你当前 CONFIG['lr'] 的设置，覆盖断点里的旧学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = CONFIG['lr']
            # ================================================================

            start_iteration = checkpoint['iteration'] + 1

            if 'history_loss' in checkpoint: history_loss = checkpoint['history_loss']

            # === 🌟 核心修复区：安全恢复历史 best_loss ===
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
            elif len(history_loss) > 0:
                # 兼容你的旧版断点：如果没存 best_loss，就从历史记录里大海捞针找出最低点
                best_loss = min(history_loss)

            print(f"恢复成功！将从第 {start_iteration} 步继续训练。")
        else:
            print("未找到断点文件，将从头开始训练。")

    model.train()
    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()
        z, log_p_z = prior.sample(CONFIG['batch_size'])
        phi, log_det_J = model(z, iteration)
        loss = torch.mean((log_p_z - log_det_J) + compute_action(phi))
        loss.backward()
        # 🌟 新增这一行：限制梯度的最大范数，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None: scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)

        if iteration % 100 == 0:
            print(
                f"迭代 {iteration:6d}/{CONFIG['iterations']} | Loss: {loss.item():.4f} | Best: {best_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': loss_val,
                        'best_loss': best_loss, 'history_loss': history_loss}, checkpoint_path)

        if loss_val < best_loss and iteration >= 10000:
            best_loss = loss_val
            print(f"🌟 第 {iteration}步发现更优模型！当前最佳 Loss: {best_loss:.4f}，正在更新 best 断点...")
            torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': best_loss,
                        'best_loss': best_loss, 'history_loss': history_loss}, save_path)

        if iteration % 1000 == 0:
            np.save(loss_save_path, np.array(history_loss))
            if iteration % 5000 == 0:
                # 兼容里程碑文件名的生成
                k_str_in_loop = f"multi_kernels_{'_'.join(map(str, CONFIG['multi_kernel_sizes']))}" if CONFIG.get(
                    'use_multi_kernel') else f"kernel_size_{CONFIG['kernel_size']}"
                milestone_path = f"prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_{k_str_in_loop}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{iteration}.pt"
                torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'loss': loss_val,
                            'best_loss': best_loss, 'history_loss': history_loss}, milestone_path)
    print("训练结束！")

if __name__ == "__main__":
    if CONFIG.get('double precision', False): torch.set_default_dtype(torch.float64)
    train()