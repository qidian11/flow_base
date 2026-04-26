import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import glob

device = torch.device(
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"运行设备: {device}")


# ==========================================
# 1. 自动化断点搜索函数 (适配动态配置)
# ==========================================
def auto_find_latest_checkpoint(config):
    """自动搜索当前配置对应的最新断点文件"""
    search_pattern = (
        f"latest_cnn_res_block_mask_model_double_precision_*_L_{config['L']}_"
        f"blockNum_{config['block_num']}_coupling_{config['coupling_layers']}_"
        f"hidden_{config['hidden_layers']}_channels_{config['hidden_channels']}_"
        f"iterations_*.pt"
    )
    matched_files = glob.glob(search_pattern)
    if not matched_files:
        return None
    return max(matched_files, key=os.path.getmtime)


# ==========================================
# 2. 标量场理论的 Action 计算
# ==========================================
def compute_action(phi, config):
    phi_up = torch.roll(phi, shifts=-1, dims=2)
    phi_down = torch.roll(phi, shifts=1, dims=2)
    phi_left = torch.roll(phi, shifts=-1, dims=3)
    phi_right = torch.roll(phi, shifts=1, dims=3)

    laplacian = 4 * phi - phi_up - phi_down - phi_left - phi_right
    action_density = phi * laplacian + config['m_sq'] * (phi ** 2) + config['lam'] * (phi ** 4)
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 动态掩码生成与卷积网络
# ==========================================
def create_dynamic_block_mask(L, block_num):
    if L % block_num != 0:
        raise ValueError(f"晶格大小 L ({L}) 必须能被 block_num ({block_num}) 整除！")
    block_size = L // block_num
    indices = torch.arange(L)
    block_indices = indices // block_size
    mask_2d = (block_indices[:, None] + block_indices[None, :]) % 2 == 0
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


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
        layers.append(nn.LeakyReLU(0.01))

        for _ in range(num_hidden_layers):
            layers.append(ResBlock(hidden_channels))

        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
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
# 4. 流模型定义 (修复数学公式版)
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, L, block_num, coupling_layers=12, hidden_channels=16, num_hidden_layers=12):
        super().__init__()
        self.L = L
        self.coupling_layers = coupling_layers
        self.register_buffer('base_mask', create_dynamic_block_mask(L, block_num))
        self.context_nets = nn.ModuleList(
            [ConvContextNet(hidden_channels=hidden_channels, num_hidden_layers=num_hidden_layers)
             for _ in range(coupling_layers)])

    def forward(self, z):
        phi = z
        log_det_jacobian = 0

        for i in range(self.coupling_layers):
            current_mask = self.base_mask if i % 2 == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi

            st_out = self.context_nets[i](phi_frozen)
            s_out = torch.tanh(st_out[:, 0:1, :, :])
            t_out = st_out[:, 1:2, :, :]

            update_mask = 1.0 - current_mask

            # 【已修复】恢复为标准仿射变换，确保数学严密与网络不崩溃
            phi_updated = update_mask * (phi * torch.exp(s_out) + t_out)
            phi = phi_frozen + phi_updated

            # 【已修复】正确的雅可比对数计算
            log_det_jacobian += torch.sum(update_mask * s_out, dim=(1, 2, 3))

        log_r_z = torch.sum(-0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi), dim=(1, 2, 3))
        log_q = log_r_z - log_det_jacobian

        return phi, log_q


# ==========================================
# 5. 独立域自训练函数
# ==========================================
def train(config, resume=True):
    L = config['L']
    block_num = config['block_num']

    # 动态生成本轮次的文件路径
    base_name = (f"cnn_res_block_mask_model_double_precision_{config['double precision']}_"
                 f"L_{L}_blockNum_{block_num}_coupling_{config['coupling_layers']}_"
                 f"hidden_{config['hidden_layers']}_channels_{config['hidden_channels']}_"
                 f"iterations_{config['iterations']}")

    save_path = f"best_{base_name}.pt"
    checkpoint_path = f"latest_{base_name}.pt"
    loss_save_path = f"{base_name}_loss_history.npy"

    model = FlowModel(L=L, block_num=block_num,
                      coupling_layers=config['coupling_layers'],
                      hidden_channels=config['hidden_channels'],
                      num_hidden_layers=config['hidden_layers']).to(device)

    if config['double precision']:
        model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['iterations'], eta_min=1e-5)

    history_loss = []
    best_loss = float('inf')
    start_iteration = 1
    is_finetuning = False

    if resume:
        old_checkpoint = auto_find_latest_checkpoint(config)
        if old_checkpoint and os.path.exists(old_checkpoint):
            print(f"🤖 检测到对应的断点文件：\n-> {old_checkpoint}\n正在恢复...")
            checkpoint = torch.load(old_checkpoint, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if config.get('double precision', False):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.double()
                is_finetuning = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-7

            start_iteration = checkpoint['iteration'] + 1

            if not is_finetuning and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'history_loss' in checkpoint:
                history_loss = checkpoint['history_loss']
                print(f"✅ 成功恢复历史 Loss，当前有 {len(history_loss)} 条数据。")
            elif os.path.exists(loss_save_path):
                history_loss = list(np.load(loss_save_path))
                print(f"✅ 已从 npy 文件恢复历史 Loss，当前有 {len(history_loss)} 条数据。")

            print(f"🚀 将从第 {start_iteration} 步继续训练至 {config['iterations']} 步。")
        else:
            print(f"🔍 未找到 block_num={block_num} 的断点，将从头开始训练。")

    model.train()

    for iteration in range(start_iteration, config['iterations'] + 1):
        optimizer.zero_grad()

        z = torch.randn(config['batch_size'], 1, L, L, device=device)
        phi, log_q = model(z)
        S_phi = compute_action(phi, config)
        loss = torch.mean(log_q + S_phi)

        loss.backward()
        optimizer.step()

        if not is_finetuning:
            scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)

        if iteration % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"迭代 {iteration:6d}/{config['iterations']} | Loss: {loss_val:.4f} | LR: {current_lr:.2e}")

            save_dict = {
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
                'history_loss': history_loss,
            }
            if not is_finetuning:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict, checkpoint_path)

        if loss_val < best_loss and iteration >= max(5000, config['iterations'] // 3):
            best_loss = loss_val
            torch.save(model.state_dict(), save_path)

        if iteration % 1000 == 0:
            np.save(loss_save_path, np.array(history_loss))

    print(f"🎉 block_num={block_num} 训练结束！\n")


# ==========================================
# 6. 自动化批处理执行入口
# ==========================================
if __name__ == "__main__":
    # 基础物理配置
    BASE_CONFIG = {
        'L': 8,
        'batch_size': 1024,
        'lr': 1e-4,
        'm_sq': -4.0,
        'lam': 6.008,
        'iterations': 15000,
        'coupling_layers': 10,
        'hidden_layers': 6,
        'hidden_channels': 16,
        'double precision': False,
    }

    if BASE_CONFIG.get('double precision', False):
        torch.set_default_dtype(torch.float64)

    # 想要测试的分区数量列表
    block_nums_to_test = [2, 4, 8]

    for b_num in block_nums_to_test:
        print("=" * 60)
        print(f"🚀 开始执行实验对照组: block_num = {b_num} (晶格 L={BASE_CONFIG['L']})")
        print("=" * 60)

        # 复制基础配置并注入当前的 block_num
        current_config = BASE_CONFIG.copy()
        current_config['block_num'] = b_num

        # 执行独立训练
        train(config=current_config, resume=True)