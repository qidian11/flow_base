import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import glob

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
# 1. 自动化断点搜索函数 (适配动态配置与Prior)
# ==========================================
def is_valid_checkpoint(filepath):
    try:
        torch.load(filepath, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def auto_find_latest_checkpoint(config):
    """自动搜索当前配置对应的最新断点文件 (包含 Prior 和 block_num 前缀)"""
    base_pattern = (
        f"prior_cnn_res_block_mask_double_precision_*_L_{config['L']}_"
        f"blockNum_{config['block_num']}_coupling_{config['coupling_layers']}_"
        f"hidden_{config['hidden_layers']}_channels_{config['hidden_channels']}_"
        f"iterations_*.pt"
    )

    def get_sorted_files(prefix):
        files = glob.glob(prefix + base_pattern)
        return sorted(files, key=os.path.getmtime, reverse=True)

    for file in get_sorted_files("latest_"):
        if is_valid_checkpoint(file):
            return file
        print(f"⚠️ 警告: 检测到损坏的 latest 断点并已自动跳过 -> {file}")

    for file in get_sorted_files("best_"):
        if is_valid_checkpoint(file):
            print(f"🔄 未找到可用的 latest，已自动回退到最新的 best 断点 -> {file}")
            return file
        print(f"⚠️ 警告: 检测到损坏的 best 断点并已自动跳过 -> {file}")

    for file in get_sorted_files(""):
        filename = os.path.basename(file)
        if not filename.startswith(("latest_", "best_")) and is_valid_checkpoint(file):
            print(f"🔄 best 也不可用，已自动回退到里程碑断点 -> {file}")
            return file

    return None


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
# 3. 动态掩码生成与卷积网络 (CNN ResBlock Mask 逻辑)
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
# 4. 自由场先验 (Free Field Prior 逻辑)
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

        log_det_factor = 0.5 * torch.sum(torch.log(2.0 * K))
        self.register_buffer('log_det_factor', log_det_factor)

        const_factor = 0.5 * self.V * math.log(2.0 * math.pi)
        self.register_buffer('const_factor', torch.tensor(const_factor))

    def sample(self, batch_size):
        eta = torch.randn(batch_size, 1, self.L, self.L, device=self.sqrt_2K.device, dtype=self.sqrt_2K.dtype)
        eta_k = torch.fft.fftn(eta, dim=(-2, -1), norm="ortho")
        phi_k = eta_k / self.sqrt_2K
        phi_free = torch.fft.ifftn(phi_k, dim=(-2, -1), norm="ortho").real

        log_p_eta = -0.5 * torch.sum(eta ** 2, dim=(1, 2, 3)) - self.const_factor
        log_p_phi = log_p_eta + self.log_det_factor
        return phi_free, log_p_phi


# ==========================================
# 5. 流模型定义 (仅负责非线性双射与雅可比行列式)
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
        """
        接收来自 FreeFieldPrior 的自由场 z
        返回变形后的 phi 以及雅可比行列式的对数
        """
        phi = z
        log_det_jacobian = 0

        for i in range(self.coupling_layers):
            current_mask = self.base_mask if i % 2 == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi

            st_out = self.context_nets[i](phi_frozen)
            s_out = torch.tanh(st_out[:, 0:1, :, :])
            t_out = st_out[:, 1:2, :, :]

            update_mask = 1.0 - current_mask
            phi_updated = update_mask * (phi * torch.exp(s_out) + t_out)
            phi = phi_frozen + phi_updated

            log_det_jacobian += torch.sum(update_mask * s_out, dim=(1, 2, 3))

        return phi, log_det_jacobian


# ==========================================
# 6. 独立域自训练函数 (结合 Prior 损失计算)
# ==========================================
def train(config, resume=True):
    L = config['L']
    block_num = config['block_num']

    # 文件命名加入 prior 标识符
    base_name = (f"prior_cnn_res_block_mask_double_precision_{config['double precision']}_"
                 f"L_{L}_blockNum_{block_num}_coupling_{config['coupling_layers']}_"
                 f"hidden_{config['hidden_layers']}_channels_{config['hidden_channels']}_"
                 f"iterations_{config['iterations']}")

    save_path = f"best_{base_name}.pt"
    checkpoint_path = f"latest_{base_name}.pt"
    loss_save_path = f"{base_name}_loss_history.npy"

    # 初始化流模型
    model = FlowModel(L=L, block_num=block_num,
                      coupling_layers=config['coupling_layers'],
                      hidden_channels=config['hidden_channels'],
                      num_hidden_layers=config['hidden_layers']).to(device)

    # 初始化自由场先验 (破缺相取正质量)
    prior_m_sq = abs(config['m_sq'])
    prior = FreeFieldPrior(L=L, m_sq_prior=prior_m_sq).to(device)

    if config['double precision']:
        model = model.double()
        prior = prior.double()

    # 尝试开启 PyTorch 编译加速
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("🚀 正在尝试开启 torch.compile() 加速...")
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['iterations'], eta_min=1e-5)

    history_loss = []
    best_loss = float('inf')
    start_iteration = 1

    if resume:
        old_checkpoint = auto_find_latest_checkpoint(config)
        if old_checkpoint and os.path.exists(old_checkpoint):
            print(f"🤖 检测到断点文件：\n-> {old_checkpoint}\n正在恢复...")
            checkpoint = torch.load(old_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if config.get('double precision', False):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.double()

            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr']

            start_iteration = checkpoint['iteration'] + 1

            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'history_loss' in checkpoint:
                history_loss = checkpoint['history_loss']
                print(f"✅ 成功恢复历史 Loss，当前有 {len(history_loss)} 条数据。")

            print(f"🚀 将从第 {start_iteration} 步继续训练至 {config['iterations']} 步。")
        else:
            print(f"🔍 未找到 block_num={block_num} 的断点，将从头开始基于先验进行训练。")

    model.train()

    for iteration in range(start_iteration, config['iterations'] + 1):
        optimizer.zero_grad()

        # 1. 结合 Prior: 从自由场中采样出基础变量 z 及其自身的对数概率 log_p_z
        z, log_p_z = prior.sample(config['batch_size'])

        # 2. 流模型仅负责非线性变形映射和雅可比计算
        phi, log_det_J = model(z)

        # 3. 联合计算生成场 phi 的概率密度
        log_q = log_p_z - log_det_J

        # 4. 计算目标作用量 S_phi，进行 KL 散度下降
        S_phi = compute_action(phi, config)
        loss = torch.mean(log_q + S_phi)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)

        if iteration % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"迭代 {iteration:6d}/{config['iterations']} | Loss: {loss_val:.4f} | LR: {current_lr:.2e}")

            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_val,
                'history_loss': history_loss,
            }, checkpoint_path)

        # if loss_val < best_loss and iteration >= max(5000, config['iterations'] // 3):
        #     best_loss = loss_val
        #     print(f"🌟 第 {iteration}步发现当前组更优模型！Loss: {best_loss:.4f}，更新 best 断点...")
        #     torch.save({
        #         'iteration': iteration,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'loss': best_loss,
        #         'history_loss': history_loss,
        #     }, save_path)

        if iteration % 1000 == 0:
            np.save(loss_save_path, np.array(history_loss))

        # 里程碑保存
        if iteration % 5000 == 0:
            milestone_path = f"prior_cnn_res_block_mask_double_precision_{config['double precision']}_L_{L}_blockNum_{block_num}_coupling_{config['coupling_layers']}_hidden_{config['hidden_layers']}_channels_{config['hidden_channels']}_iterations_{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_val,
                'history_loss': history_loss,
            }, milestone_path)

    print(f"🎉 block_num={block_num} 训练结束！\n")


# ==========================================
# 7. 自动化批处理执行入口
# ==========================================
if __name__ == "__main__":
    # 基础物理配置 (与 cnn_res_block_mask 保持一致)
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
        print('已开启全局双精度。')

    # 想要测试的分区数量列表
    block_nums_to_test = [2, 4, 8]

    for b_num in block_nums_to_test:
        print("=" * 60)
        print(f"🚀 开始执行 Prior 组实验: block_num = {b_num} (晶格 L={BASE_CONFIG['L']})")
        print("=" * 60)

        # 复制基础配置并注入当前的 block_num
        current_config = BASE_CONFIG.copy()
        current_config['block_num'] = b_num

        # 执行独立训练
        train(config=current_config, resume=True)