import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pyglm.glm import mvec2

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Phi4Action(nn.Module):
    def __init__(self, m2, lam, L):
        super(Phi4Action, self).__init__()
        self.m2 = m2
        self.lam = lam
        self.L = L

    def forward(self, phi):
        # reshape to(Batch, L, L)
        phi = phi.view(-1, self.L, self.L)
        # 向右移动 (x+1)
        phi_right = torch.roll(phi, shifts=-1, dims=2)
        # 向下移动 (y+1)
        phi_down = torch.roll(phi, shifts=-1, dims=1)

        # 计算差分平方
        # sum_mu (phi(x) - phi(x+mu))^2
        # 我们只需要计算正方向的差分并求和，因为遍历所有 x 会覆盖所有连接
        delta_x_sq = (phi - phi_right) ** 2
        delta_y_sq = (phi - phi_down) ** 2

        kinetic_term = torch.sum(delta_x_sq + delta_y_sq, dim=(1, 2))

        # --- 2. 质量项 (Mass Term) ---
        # 论文公式 (20) 第二项: m^2 * phi^2
        # 注意：这里没有 0.5 的系数！且 m^2 在论文中取值为 -4.0
        mass_term = self.m2 * torch.sum(phi ** 2, dim=(1, 2))

        # --- 3. 相互作用项 (Interaction Term) ---
        # 论文公式 (20) 第三项: lambda * phi^4
        # 注意：这里没有 1/4 或 1/24 的系数！
        interaction_term = self.lam * torch.sum(phi ** 4, dim=(1, 2))

        # 总作用量
        S = kinetic_term + mass_term + interaction_term
        return S

class ScaleTranslationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 论文 Section III A: "two to six fully-connected layers... leaky rectified linear unit"
        # 这里我们选择 3 层作为示例，hidden_dim 设为 256 或 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # Leaky ReLU slope 通常取 0.01-0.2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2 * input_dim)  # 输出拼接的 s 和 t，维度是输入的两倍
        )

        # 初始化技巧 (虽然论文没详述，但这在 Flow 模型中是标准操作)：
        # 将最后一层的权重和偏置初始化为 0。
        # 这样初始时 s=0, t=0，模型一开始就是恒等变换 (Identity)。
        # 这能极大地帮助模型从一开始的“高斯噪声”平稳过渡到训练状态。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # 输入 x 是被 Mask 遮挡后的 z_a
        out = self.net(x)
        s, t = out.chunk(2, dim=1)  # 切分为 s 和 t

        # 对 s 进行 Tanh 约束是一个常见的工程技巧，防止缩放因子爆炸
        # 但为了严格复现论文，论文公式 (9) 直接用 e^s，未提及 Tanh。
        # 我们这里保持原始实现。如果训练不稳定，可以考虑 s = torch.tanh(s)
        return s, t

def make_checkerboard_mask(L, parity):
    """
    创建棋盘格掩码。
    L: 格子边长
    parity: 0 或 1，控制掩码的相位（先更新黑格还是白格）
    """
    checkerboard = np.indices((L, L)).sum(axis=0) % 2
    mask = (checkerboard + parity) % 2
    # 展平为向量，因为我们的全连接网络接受的是 flat input
    return torch.from_numpy(mask.flatten()).float()


class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super().__init__()
        self.input_dim = input_dim
        # 将 mask 注册为 buffer，这样它会随模型保存，但不是训练参数
        self.register_buffer('mask', mask)

        # s 和 t 网络
        # 注意：实际上 s/t 网络只需要处理 masked input，
        # 但为了实现简单，通常输入整个向量，mask 在内部应用。
        self.st_net = ScaleTranslationNet(input_dim, hidden_dim)

    def forward(self, z):
        """
        执行逆变换 z -> phi (生成方向)
        """
        # 1. 区分 z_a (保持不变) 和 z_b (需要变换)
        # mask 为 1 的位置是 z_a
        z_a = z * self.mask

        # 2. 通过神经网络计算 s 和 t
        # 网络只应该“看到” z_a，所以输入也是 masked 的
        s, t = self.st_net(z_a)

        # 3. 确保 s 和 t 只作用在 z_b 部分 (mask 为 0 的部分)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # 4. 执行变换公式 (10)
        # phi_a = z_a
        # phi_b = (z_b - t) * exp(-s)
        phi = z_a + (1 - self.mask) * (z - t) * torch.exp(-s)

        # 5. 计算这一层的 Log Determinant
        # 雅可比矩阵是对角阵，元素为 exp(-s)
        # log_det = sum(log(exp(-s))) = sum(-s)
        log_det_J = torch.sum(-s, dim=1)

        return phi, log_det_J


class RealNVP(nn.Module):
    def __init__(self, L, hidden_dim, n_layers=8):
        super().__init__()
        self.L = L
        self.input_dim = L * L

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # 交替掩码相位：一层偶数点，一层奇数点
            mask = make_checkerboard_mask(L, parity=(i % 2))
            layer = AffineCouplingLayer(self.input_dim, hidden_dim, mask)
            self.layers.append(layer)

    def sample(self, batch_size, device='cpu'):
        """
        生成样本 phi，并返回其对应的 log probability
        用于计算损失函数
        """
        # 1. 从先验分布 r(z) 采样 (标准正态分布)
        #  r(z) propto exp(-z^2/2)
        z = torch.randn(batch_size, self.input_dim).to(device)

        # 计算先验的 log probability: log r(z)
        # log(1/sqrt(2pi) * exp(-z^2/2)) = -0.5*z^2 - 0.5*log(2pi)
        log_prob_z = -0.5 * torch.sum(z ** 2, dim=1) - \
                     0.5 * self.input_dim * np.log(2 * np.pi)

        x = z
        sum_log_det = 0

        # 2. 依次通过流层 (Flow Layers)
        for layer in self.layers:
            x, log_det = layer(x)
            sum_log_det += log_det

        # 3. 计算最终的 log q(phi)
        # 变量代换公式: log q(x) = log r(z) - log |det dz/dx|
        # 由于我们在 layer 中计算的是 log |det dx/dz| (即生成方向的雅可比)
        # 关系是: log |det dz/dx| = - log |det dx/dz|
        # 所以: log q(x) = log r(z) - (- sum_log_det) = log r(z) + sum_log_det
        # 等等，我们需要小心符号。
        # 公式 (8): p(phi) = r(z) * |det df/dphi|
        # 这里的 f 是 phi -> z.
        # 我们 layer 输出的是 phi = g(z). 所以 z = g^-1(phi).
        # Jacobian 应该是 |det d(g^-1)/dphi| = |det dz/dphi|.
        # 我们代码里 layer 算的是 log_det_J = log |det dphi/dz|.
        # 所以 log |det dz/dphi| = - log_det_J.
        #
        # log p(phi) = log r(z) + log |det dz/dphi|
        #            = log r(z) - log_det_J_total

        log_prob_x = log_prob_z - sum_log_det

        return x, log_prob_x


# 设置参数 (严格按照 Table I, L=6 的情况作为快速演示)
# L=6, m^2=-4.0, lambda=6.975 [cite: 203]
L = 6
m2 = -4.0
lam = 6.975

# 实例化“考卷” (Step 1)
action_fn = Phi4Action(m2=m2, lam=lam, L=L).to(device)

# 实例化“学生” (Step 2)
# input_dim = L^2, hidden_dim 可以设大一点比如 256
model = RealNVP(L=L, hidden_dim=256, n_layers=8).to(device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 2. 训练循环 ---
batch_size = 1024  # M in Eq. (14)
n_epochs = 2000
loss_history = []

print(f"开始在 {device} 上训练...")

for epoch in range(n_epochs):
    # [关键步骤 A] 采样
    # 对应数学：phi ~ tilde{p}_f
    # 这一步，model 凭空生成了 M 个样本，并告诉我们它们在当前模型下的概率
    phi, log_prob_model = model.sample(batch_size, device=device)

    # [关键步骤 B] 计算物理作用量
    # 对应数学：S(phi)
    # 这一步，我们用“物理定律”来给这些样本打分
    action_value = action_fn(phi)

    # [关键步骤 C] 计算损失函数
    # 对应公式 (14): L = mean( log_p + S )
    # 注意：论文里的 Loss 是 Shifted KL，不需要计算配分函数 Z
    loss = torch.mean(log_prob_model + action_value)

    # [关键步骤 D] 反向传播更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录 Loss
    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

print("训练完成！")

# --- 3. 可视化 Loss ---
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss (Shifted KL)")
plt.title(f"Training Loss (L={L})")
plt.show()