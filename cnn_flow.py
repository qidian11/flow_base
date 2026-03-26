import torch
import torch.nn as nn
import torch.optim as optim
import math

device = torch.device(
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"运行设备: {device}")

# ==========================================
# 1. 物理参数配置 (严格对齐论文参数)
# ==========================================
CONFIG = {
    'L': 14,  # 晶格大小 (对应实验 E5)
    'm_sq': -4.0,  # m^2 (质量的平方)
    'lam': 5.113,  # lambda (耦合常数)
    'batch_size': 1024,  # 批大小 [cite: 2613]
    'lr': 1e-3,  # 学习率 [cite: 2614]
    'iterations': 100000  # 训练迭代次数 [cite: 2615]
}


# ==========================================
# 2. 标量场理论的 Action 计算 (物理约束)
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


class ConvContextNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 结构: 单通道输入, 2层隐藏层(各8通道), 3x3卷积核, stride=1, 周期性填充 [cite: 2601, 2602, 2603]
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.LeakyReLU(0.01),  # f(x) = max(0, x) + 0.01*min(0, x) [cite: 2604]
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.Tanh()  # 限制输出范围避免 e^s 导致数值爆炸 [cite: 2605]
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 4. 流模型定义 (完全生成模式)
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, L, coupling_layers=12):  # 12个耦合层 [cite: 2587]
        super().__init__()
        self.L = L
        self.coupling_layers = coupling_layers
        self.register_buffer('base_mask', create_checkerboard_mask(L))
        self.context_nets = nn.ModuleList([ConvContextNet() for _ in range(coupling_layers)])

    def forward(self, z):
        """
        前向生成过程: z (先验噪声) -> phi (物理场构型) [cite: 2987, 3076]
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
            s_out = st_out[:, 0:1, :, :]
            t_out = st_out[:, 1:2, :, :]

            # 需要更新的区域
            update_mask = 1.0 - current_mask

            # 论文仿射变换公式: m*phi + (1-m) * (e^s * phi + t) [cite: 2593]
            # 注意：此处我们在做生成 (z -> phi)
            phi_updated = update_mask * (phi * torch.exp(s_out) + t_out)
            phi = phi_frozen + phi_updated

            # 累加雅可比行列式的对数
            log_det_jacobian += torch.sum(update_mask * s_out, dim=(1, 2, 3))

        # 1. 计算先验概率对数 log r(z) (独立同分布的标准正态分布) [cite: 2580, 3273]
        log_r_z = torch.sum(-0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi), dim=(1, 2, 3))

        # 2. 计算输出概率密度 log q(phi) = log r(z) - log|det(d_phi / d_z)| [cite: 3082, 3109]
        log_q = log_r_z - log_det_jacobian

        return phi, log_q


# ==========================================
# 5. 自训练循环 (不依赖外部数据)
# ==========================================
def train():
    L = CONFIG['L']
    model = FlowModel(L=L).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])  # [cite: 2614]

    model.train()
    print("开始自训练...")

    for iteration in range(1, CONFIG['iterations'] + 1):
        optimizer.zero_grad()

        # 1. 从先验分布(噪声)中纯随机采样 [cite: 2580]
        z = torch.randn(CONFIG['batch_size'], 1, L, L, device=device)

        # 2. 生成物理场 phi 并计算其生成的概率对数 log q(phi)
        phi, log_q = model(z)

        # 3. 计算生成的场在物理理论下的 Action S(phi)
        S_phi = compute_action(phi)

        # 4. 计算 KL 散度损失: Loss = mean(log q(phi) + S(phi))
        loss = torch.mean(log_q + S_phi)

        loss.backward()
        optimizer.step()

        # 打印日志 (为了方便观察，你可以自己调节频率)
        if iteration % 100 == 0:
            print(f"迭代 {iteration:6d}/{CONFIG['iterations']} | Loss: {loss.item():.4f}")

    print("100,000 次自训练完成！现在你可以用这个模型来采样物理场了。")


if __name__ == "__main__":
    train()