import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

device = torch.device(
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"运行设备: {device}")

# ==========================================
# 1. 物理参数配置
# ==========================================
CONFIG = {
    'L': 14,  # 晶格大小 (对应实验 E5)
    'm_sq': -4.0,  # m^2 (质量的平方)
    'lam': 5.113,  # lambda (耦合常数)
    'batch_size': 1024,  # 批大小
    'lr': 1e-3,  # 学习率
    'iterations': 55000,  # 训练迭代次数
    'coupling_layers': 16,
    'hidden_layers': 6, # 实际上是6*3
    'hidden_channels': 16,
    'double precision': True,
}
save_path = f"best_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
loss_save_path = f"cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_loss_history_coupling_layers_{CONFIG['coupling_layers']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npy"
checkpoint_path = f"latest_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
phi_ensemble_save_path = f"phi_ensemble_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_loss_history_coupling_layers_{CONFIG['coupling_layers']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npz"
CONFIG['save_path'] = save_path
CONFIG['loss_save_path'] = loss_save_path
CONFIG['checkpoint_path'] = checkpoint_path
CONFIG['phi_ensemble_save_path'] = phi_ensemble_save_path


# ==========================================
# 2. 标量场理论的 Action 计算
# ==========================================
def compute_action(phi):
    """
    计算 (1+1)D phi^4 标量场理论的晶格作用量 S(phi)
    phi 形状: (Batch, 1, L, L)
    公式: S = sum_x [ phi(x)*Laplacian(phi)(x) + m^2*phi(x)^2 + lambda*phi(x)^4 ] [cite: 2501, 3180]
    """
    # 周期性边界条件下的位移
    phi_up = torch.roll(phi, shifts=-1, dims=2)
    phi_down = torch.roll(phi, shifts=1, dims=2)
    phi_left = torch.roll(phi, shifts=-1, dims=3)
    phi_right = torch.roll(phi, shifts=1, dims=3)

    # 计算 d'Alembert/Laplace 算子: sum_mu (2*phi(x) - phi(x-mu) - phi(x+mu))
    laplacian = 4 * phi - phi_up - phi_down - phi_left - phi_right

    # 计算局部的能量密度
    action_density = phi * laplacian + CONFIG['m_sq'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)

    # 沿着空间维度求和，得到每个样本的总 Action
    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 掩码生成与卷积上下文网络
# ==========================================
def create_checkerboard_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()


# 【新增】定义一个标准的残差块
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 保持通道数不变的两次卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.act1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.act2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.act3 = nn.LeakyReLU(0.01)

    def forward(self, x):
        # 残差连接加在这里！只在同等维度的隐藏特征之间相加
        return x + self.act3(self.conv3(self.act2(self.conv2(self.act1(self.conv1(x))))))


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4):
        super().__init__()
        layers = []

        # 1. 升维映射：1 通道 -> hidden_channels (把单通道的物理场升维成多通道特征)
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
        layers.append(nn.LeakyReLU(0.01))

        # 2. 预设数量的残差块 (特征提取阶段，疯狂堆叠深度，且不会梯度消失)
        for _ in range(num_hidden_layers):
            layers.append(ResBlock(hidden_channels))

        # 3. 输出层：hidden_channels -> 2 通道 (把多通道特征降维成我们需要的 s 和 t)
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, stride=1, padding=1, padding_mode='circular'))

        # 自动构建 Sequential
        self.net = nn.Sequential(*layers)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

        # 【极其关键】强制最后一层零初始化！防止深层网络起步爆炸
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 4. 流模型定义 (完全生成模式)
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, L, coupling_layers=12, hidden_channels=16, num_hidden_layers=12):  # 默认12个耦合层
        super().__init__()
        self.L = L
        self.coupling_layers = coupling_layers
        self.register_buffer('base_mask', create_checkerboard_mask(L))
        self.context_nets = nn.ModuleList(
            [ConvContextNet(hidden_channels=hidden_channels,
                            num_hidden_layers=num_hidden_layers)
             for _ in range(coupling_layers)])

    def forward(self, z):
        """
        前向生成过程: z (先验噪声) -> phi (物理场构型)
        返回:
            phi: 生成的场
            log_q: 生成该场的概率密度的对数 log q(phi)
        """
        phi = z
        # 记录映射的 Log-Jacobian 行列式
        log_det_jacobian = 0

        for i in range(self.coupling_layers):
            current_mask = self.base_mask if i % 2 == 0 else (1.0 - self.base_mask)

            # 被冻结的部分
            phi_frozen = current_mask * phi

            # 经过 CNN 计算 s 和 t
            st_out = self.context_nets[i](phi_frozen)
            # 对 s_out 使用 tanh 限制范围防爆炸，t_out 保持线性
            s_out = torch.tanh(st_out[:, 0:1, :, :])
            t_out = st_out[:, 1:2, :, :]

            # 需要更新的区域
            update_mask = 1.0 - current_mask

            # 论文仿射变换公式: m*phi + (1-m) * (e^s * phi + t)
            # 注意：此处我们在做生成 (z -> phi)
            phi_updated = update_mask * (phi * torch.exp(s_out) + t_out)
            phi = phi_frozen + phi_updated

            # 累加雅可比行列式的对数
            log_det_jacobian += torch.sum(update_mask * s_out, dim=(1, 2, 3))

        # 1. 计算先验概率对数 log r(z) (独立同分布的标准正态分布)
        log_r_z = torch.sum(-0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi), dim=(1, 2, 3))

        # 2. 计算输出概率密度 log q(phi) = log r(z) - log|det(d_phi / d_z)|
        log_q = log_r_z - log_det_jacobian

        return phi, log_q


# ==========================================
# 5. 自训练循环 (支持断点续训)
# ==========================================
def train(save_path=CONFIG['save_path'], loss_save_path=CONFIG['loss_save_path'], resume=True, checkpoint_path=CONFIG['checkpoint_path']):
    L = CONFIG['L']
    model = FlowModel(L=L, coupling_layers=CONFIG['coupling_layers'],
                      hidden_channels=CONFIG['hidden_channels'],
                      num_hidden_layers=CONFIG['hidden_layers']).to(device)
    if CONFIG['double precision']:
        # 显式转换为双精度
        model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # 定义余弦退火学习率调度器 (T_max 依然是配置中的总步数)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=CONFIG['iterations'], eta_min=1e-5
    # )

    history_loss = []
    best_loss = float('inf')
    start_iteration = 1

    # ================= 新增：断点恢复逻辑 =================
    if resume:
        import os
        old_checkpoint_path = "latest_cnn_res_model_14_coupling_layers_16_hidden_layers_6_hidden_channels_16_iterations_50000.pt"
        old_loss_path = 'cnn_res_model_14_loss_history_coupling_layers_16_hidden_layers_6_hidden_channels_16_iterations_50000.npy'
        if os.path.exists(old_checkpoint_path):
            print(f"检测到断点文件，正在从 {old_checkpoint_path} 恢复训练...")
            checkpoint = torch.load(old_checkpoint_path, map_location=device)

            # 恢复模型和优化器状态
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # ================= 强制对齐优化器内部状态的精度 =================
            if CONFIG.get('double precision', False):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        # 如果状态是张量，强转为双精度 (FP64)
                        if isinstance(v, torch.Tensor):
                            state[k] = v.double()

            # 锁定在 1e-6 进行极致微调
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-7  # 锁定在 1e-6 进行极致微调

            # 获取上次中断的步数
            start_iteration = checkpoint['iteration'] + 1

            # load scheduler_state_dict
            # if 'scheduler_state_dict' in checkpoint:
            #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # else:
            #     # 兼容旧版本：如果没有保存 scheduler 状态，手动空跑 step 追赶进度
            #     print("未检测到 scheduler 状态，正在同步学习率进度...")
            #     for _ in range(start_iteration - 1):
            #         scheduler.step()



            # ================= 修改：更严谨的 Loss 恢复逻辑 =================
            if 'history_loss' in checkpoint:
                # 新版本：直接从断点读取严格同步的原始 Loss 记录
                history_loss = checkpoint['history_loss']
                print(f"成功从 checkpoint 恢复历史 Loss，当前有 {len(history_loss)} 条未平滑原始数据。")
            elif os.path.exists(old_loss_path):
                # 兼容老版本断点
                history_loss = list(np.load(old_loss_path))
                print(f"未在断点中找到 Loss，已从 npy 文件恢复历史 Loss，当前有 {len(history_loss)} 条数据。")
            # ================================================================

            print(f"恢复成功！将从第 {start_iteration} 步继续训练至 {CONFIG['iterations']} 步。")
        else:
            print(f"未找到断点文件 {old_checkpoint_path}，将从头开始训练。")
    # =====================================================

    model.train()
    print("开始自训练...")

    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()

        z = torch.randn(CONFIG['batch_size'], 1, L, L, device=device)
        phi, log_q = model(z)
        S_phi = compute_action(phi)
        loss = torch.mean(log_q + S_phi)

        loss.backward()
        optimizer.step()
        # scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)

        # 打印日志与保存
        if iteration % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # print(
            #     f"迭代 {iteration:6d}/{CONFIG['iterations']} | Loss: {loss.item():.4f} | "
            #     f"LR: {scheduler.get_last_lr()[0]:.2e}")
            print(
                f"迭代 {iteration:6d}/{CONFIG['iterations']} | Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.2e}")

            # 把 scheduler 的状态也存进去，方便下次彻底恢复
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),  # 新增保存 scheduler
                'loss': loss_val,
                'history_loss': history_loss,
            }, checkpoint_path)

        if loss_val < best_loss and iteration >= 10000:
            best_loss = loss_val
            torch.save(model.state_dict(), save_path)

        if iteration % 1000 == 0:
            np.save(loss_save_path, np.array(history_loss))

    print("训练结束！")


if __name__ == "__main__":
    if CONFIG.get('double precision', False): # 开启双精度
        torch.set_default_dtype(torch.float64)
    train(CONFIG['save_path'],CONFIG['loss_save_path'])