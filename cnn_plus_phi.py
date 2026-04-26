import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import os
import glob

# 自动适配运行设备
device = torch.device(
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"运行设备: {device}")

# ==========================================
# 1. 物理参数与全局配置
# ==========================================
CONFIG = {
    'L': 14,                # 晶格大小 (对应实验 E5)
    'm_sq': -4.0,           # m^2 (质量的平方)
    'lam': 5.113,           # lambda (耦合常数)
    'batch_size': 1024,     # 批大小
    'lr': 1e-3,             # 初始学习率
    'iterations': 10000,    # 训练迭代次数
    'coupling_layers': 16,  # 流层数量
    'hidden_layers': 6,     # 残差块数量
    'hidden_channels': 16,  # 卷积通道数
    'K': 8,                 # RQS 分段样条的数量 (Bins)
    'bound': 5.0,           # RQS 场变量的物理边界范围 [-5.0, 5.0]
    'double precision': False, # 是否开启双精度 FP64 (微调阶段建议设为 True)
}

# 【更新】将 K 和 bound 注入基础文件名模板
base_name_template = (
    f"cnn_rqs_model_double_precision_{CONFIG['double precision']}_"
    f"{CONFIG['L']}_coupling_{CONFIG['coupling_layers']}_"
    f"hidden_{CONFIG['hidden_layers']}_channels_{CONFIG['hidden_channels']}_"
    f"K_{CONFIG['K']}_bound_{CONFIG['bound']}_"
    f"iterations_{CONFIG['iterations']}"
)

CONFIG['save_path'] = f"best_{base_name_template}.pt"
CONFIG['checkpoint_path'] = f"latest_{base_name_template}.pt"
CONFIG['loss_save_path'] = f"{base_name_template}_loss_history.npy"


# ==========================================
# 2. 自动化断点搜索函数 (增强版)
# ==========================================
def auto_find_latest_checkpoint(config):
    """
    自动搜索架构匹配的最新断点文件。
    1. 使用 '*' 忽略 double_precision 的差异，允许 FP64 继承 FP32 的权重。
    2. 使用 '*' 忽略 iterations 的差异，允许增加总迭代次数后继续训练。
    3. 严格匹配 L, coupling, hidden_layers, channels, K 和 bound，确保物理与网络架构对齐。
    """
    search_pattern = (
        f"latest_cnn_rqs_model_double_precision_*_"
        f"{config['L']}_coupling_{config['coupling_layers']}_"
        f"hidden_{config['hidden_layers']}_channels_{config['hidden_channels']}_"
        f"K_{config['K']}_bound_{config['bound']}_"
        f"iterations_*.pt"
    )
    matched_files = glob.glob(search_pattern)
    if not matched_files:
        return None
    # 返回修改时间最近的文件
    latest_file = max(matched_files, key=os.path.getmtime)
    return latest_file


# ==========================================
# 3. 物理 Action 计算
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
# 4. 网络架构 (RQS)
# ==========================================
def create_block_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.act1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.act2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.act3 = nn.LeakyReLU(0.01)
    def forward(self, x):
        return x + self.act3(self.conv3(self.act2(self.conv2(self.act1(self.conv1(x))))))

class ConvContextNetSpline(nn.Module):
    def __init__(self, hidden_channels=16, num_hidden_layers=6, K=8):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
        layers.append(nn.LeakyReLU(0.01))
        for _ in range(num_hidden_layers):
            layers.append(ResBlock(hidden_channels))
        out_channels = 3 * K - 1
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
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
# 5. RQS 核心函数
# ==========================================
def rational_quadratic_spline(inputs, unnorm_widths, unnorm_heights, unnorm_derivatives, bound=5.0):
    B, _, H, W = inputs.shape
    K = unnorm_widths.shape[1]
    min_bin_val = 1e-3

    widths = F.softmax(unnorm_widths, dim=1)
    widths = min_bin_val + (1 - min_bin_val * K) * widths
    heights = F.softmax(unnorm_heights, dim=1)
    heights = min_bin_val + (1 - min_bin_val * K) * heights
    derivs = F.softplus(unnorm_derivatives) + min_bin_val
    pad = torch.ones(B, 1, H, W, dtype=inputs.dtype, device=inputs.device)
    derivs = torch.cat([pad, derivs, pad], dim=1)

    cumwidths = torch.cumsum(widths, dim=1)
    cumwidths = F.pad(cumwidths, pad=(0, 0, 0, 0, 1, 0), mode='constant', value=0.0)
    cumwidths = (cumwidths * 2 * bound) - bound
    widths = widths * 2 * bound

    cumheights = torch.cumsum(heights, dim=1)
    cumheights = F.pad(cumheights, pad=(0, 0, 0, 0, 1, 0), mode='constant', value=0.0)
    cumheights = (cumheights * 2 * bound) - bound
    heights = heights * 2 * bound

    inside_mask = (inputs >= -bound) & (inputs <= bound)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)

    if not inside_mask.any():
        return outputs, logabsdet

    # 4. 核心：通过 searchsorted 寻找场变量所在的区间 Bin
    # 【修复警告】：显式调用 contiguous() 保证底层 C++ 二分查找的内存连续性
    inputs_p = inputs.permute(0, 2, 3, 1).contiguous()
    cumwidths_p = cumwidths.permute(0, 2, 3, 1).contiguous()

    # +1e-6 防止边界落在前一个 bin 的死角
    bin_idx_p = torch.searchsorted(cumwidths_p, inputs_p + 1e-6) - 1
    bin_idx_p = torch.clamp(bin_idx_p, 0, K - 1)
    bin_idx = bin_idx_p.permute(0, 3, 1, 2).contiguous()  # 养成好习惯，变回来也 contiguous 一下

    input_W = torch.gather(widths, 1, bin_idx)
    input_cumW = torch.gather(cumwidths, 1, bin_idx)
    input_H = torch.gather(heights, 1, bin_idx)
    input_cumH = torch.gather(cumheights, 1, bin_idx)
    input_D0 = torch.gather(derivs, 1, bin_idx)
    input_D1 = torch.gather(derivs, 1, bin_idx + 1)

    xi = (inputs - input_cumW) / input_W
    xi = torch.clamp(xi, 0.0, 1.0)
    xi_sq = xi * xi
    inv_xi = 1.0 - xi
    inv_xi_sq = inv_xi * inv_xi
    sk = input_H / input_W

    numerator = input_H * (sk * xi_sq + input_D0 * xi * inv_xi)
    denominator = sk + (input_D1 + input_D0 - 2 * sk) * xi * inv_xi
    outputs_inside = input_cumH + numerator / denominator

    num_deriv = sk * sk * (input_D1 * xi_sq + 2 * sk * xi * inv_xi + input_D0 * inv_xi_sq)
    logabsdet_inside = torch.log(num_deriv) - 2 * torch.log(denominator)

    outputs = torch.where(inside_mask, outputs_inside, outputs)
    logabsdet = torch.where(inside_mask, logabsdet_inside, logabsdet)

    return outputs, logabsdet


# ==========================================
# 6. 流模型定义
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, L, coupling_layers=16, hidden_channels=16, num_hidden_layers=6, K=8, bound=5.0):
        super().__init__()
        self.L = L
        self.K = K
        self.bound = bound
        self.coupling_layers = coupling_layers
        self.register_buffer('base_mask', create_block_mask(L))
        self.context_nets = nn.ModuleList(
            [ConvContextNetSpline(hidden_channels=hidden_channels, num_hidden_layers=num_hidden_layers, K=K)
             for _ in range(coupling_layers)])

    def forward(self, z):
        phi = z
        log_det_jacobian = 0
        for i in range(self.coupling_layers):
            current_mask = self.base_mask if i % 2 == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi
            spline_params = self.context_nets[i](phi_frozen)
            W_unnorm = spline_params[:, :self.K, :, :]
            H_unnorm = spline_params[:, self.K:2*self.K, :, :]
            D_unnorm = spline_params[:, 2*self.K:, :, :]
            update_mask = 1.0 - current_mask
            phi_updated, log_det_J = rational_quadratic_spline(
                inputs=phi, unnorm_widths=W_unnorm, unnorm_heights=H_unnorm, unnorm_derivatives=D_unnorm, bound=self.bound
            )
            phi = phi_frozen + phi_updated * update_mask
            log_det_jacobian += torch.sum(log_det_J * update_mask, dim=(1, 2, 3))
        log_r_z = torch.sum(-0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi), dim=(1, 2, 3))
        log_q = log_r_z - log_det_jacobian
        return phi, log_q


# ==========================================
# 7. 训练循环
# ==========================================
def train(resume=True):
    L = CONFIG['L']
    model = FlowModel(L=L, coupling_layers=CONFIG['coupling_layers'],
                      hidden_channels=CONFIG['hidden_channels'],
                      num_hidden_layers=CONFIG['hidden_layers'],
                      K=CONFIG['K'], bound=CONFIG['bound']).to(device)

    if CONFIG['double precision']:
        model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['iterations'], eta_min=1e-5)

    history_loss = []
    best_loss = float('inf')
    start_iteration = 1
    is_finetuning = False

    if resume:
        # 使用自动化搜索函数
        old_checkpoint = auto_find_latest_checkpoint(CONFIG)
        if old_checkpoint:
            print(f"🤖 自动化断点匹配成功：\n-> {old_checkpoint}")
            checkpoint = torch.load(old_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 如果当前配置开启了双精度，则自动对齐状态并锁定极小学习率进行微调
            if CONFIG.get('double precision', False):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.double()
                is_finetuning = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-7
                print("💎 检测到双精度微调配置：已将参数转为 FP64 并锁定学习率至 1e-7。")

            start_iteration = checkpoint['iteration'] + 1
            if not is_finetuning and 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'history_loss' in checkpoint:
                history_loss = checkpoint['history_loss']
        else:
            print("🔍 未找到架构匹配的断点，将开始全新训练。")

    model.train()
    print(f"\n🚀 开始自训练 (L={L}, K={CONFIG['K']}, bound={CONFIG['bound']})...")

    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()
        z = torch.randn(CONFIG['batch_size'], 1, L, L, device=device)
        phi, log_q = model(z)
        S_phi = compute_action(phi)
        loss = torch.mean(log_q + S_phi)

        loss.backward()
        optimizer.step()
        if not is_finetuning:
            scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)

        if iteration % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"迭代 {iteration:6d}/{CONFIG['iterations']} | Loss: {loss_val:.4f} | LR: {current_lr:.2e}")
            save_dict = {
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
                'history_loss': history_loss,
            }
            if not is_finetuning:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(save_dict, CONFIG['checkpoint_path'])

        if loss_val < best_loss and iteration >= 5000:
            best_loss = loss_val
            torch.save(model.state_dict(), CONFIG['save_path'])

        if iteration % 1000 == 0:
            np.save(CONFIG['loss_save_path'], np.array(history_loss))

    print("\n🎉 任务完成。")

if __name__ == "__main__":
    if CONFIG.get('double precision', False):
        torch.set_default_dtype(torch.float64)
    train(resume=True)