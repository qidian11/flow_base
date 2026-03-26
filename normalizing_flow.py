import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
print(f"Running on: {device}")


# 1. 配置
CONFIG = {
    'L': 14,
    'batch_size': 1024,
    'lr': 1e-4,
    'epochs': 10000
}


# 2. 掩码生成
def create_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.flatten()  # 返回 (L*L,) 的布尔张量


# 3. 定义 s 和 t 网络
class StNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

        # 初始化权重为接近 0，从恒等变换开始
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# 4. 定义 RealNVP 模型
class RealNVPtorch(nn.Module):
    def __init__(self, L, coupling_layer_num=8, hidden_dim=128):
        super().__init__()
        self.L = L
        self.split_dim = (L * L) // 2
        self.base_mask = create_mask(L)

        self.s_nets = nn.ModuleList([StNetwork(self.split_dim, hidden_dim) for _ in range(coupling_layer_num)])
        self.t_nets = nn.ModuleList([StNetwork(self.split_dim, hidden_dim) for _ in range(coupling_layer_num)])

    def forward(self, phi):
        # phi 形状: (batch, L, L) -> (batch, L*L)
        z = phi.view(phi.size(0), -1)
        log_det_jacobian = 0

        for i in range(len(self.s_nets)):
            mask = self.base_mask if i % 2 == 0 else ~self.base_mask

            # 切分数据
            z_a = z[:, mask]
            z_b = z[:, ~mask]

            # 计算 s 和 t
            s_out = self.s_nets[i](z_a)
            t_out = self.t_nets[i](z_a)

            # 耦合变换: z_b = z_b * exp(s) + t
            z_b = z_b * torch.exp(s_out) + t_out

            # 累加 Log-Jacobian
            log_det_jacobian += torch.sum(s_out, dim=1)

            # 合并回去
            z_new = torch.empty_like(z)
            z_new[:, mask] = z_a
            z_new[:, ~mask] = z_b
            z = z_new

        return z, log_det_jacobian

    def compute_loss(self, z, log_det_jacobian):
        # Prior log-likelihood: log r(z) = sum(-0.5 * z^2)
        log_r_z = torch.sum(-0.5 * (z ** 2), dim=1)
        # Loss = -E[log r(z) + log_det_jacobian]
        loss = -torch.mean(log_r_z + log_det_jacobian)
        return loss


# 5. 模拟训练流程
def train():
    # 加载数据
    data_np = np.load("configs_L14_N12800.npy").reshape(-1, 14, 14).astype(np.float64)
    dataset = torch.from_numpy(data_np)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    model = RealNVPtorch(L=14).double()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     'min',
    #     patience=10,
    #     factor=0.9
    # )

    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            z, log_det = model(batch)
            loss = model.compute_loss(z, log_det)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        # scheduler.step(avg_loss)
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Current LR: {current_lr}")


if __name__ == "__main__":
    train()