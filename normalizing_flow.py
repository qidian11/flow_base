import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


device = torch.device(
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Running on: {device}")


# 1. 配置
CONFIG = [{
    'L': 6,
    'batch_size': 1024,
    'lr': 1e-4,
    'm2':-4,
    'lam':6.975,
    'epochs': 20000,
    'model_save_path': 'normalizing_flow_best_model_6_20000.pt',
    'loss_save_path':'normalizing_flow_loss_6_20000}.npy'
},{
    'L': 8,
    'batch_size': 1024,
    'lr': 1e-4,
    'm2':-4,
    'lam':6.008,
    'epochs': 30000,
'model_save_path': 'normalizing_flow_best_model_8_30000.pt',
    'loss_save_path':'normalizing_flow_loss_8_30000}.npy'
},{
    'L': 10,
    'batch_size': 1024,
    'lr': 1e-4,
    'm2':-4,
    'lam':5.550,
    'epochs': 50000,
'model_save_path': 'normalizing_flow_best_model_10_50000.pt',
    'loss_save_path':'normalizing_flow_loss_10_50000}.npy'
},{
    'L': 12,
    'batch_size': 1024,
    'lr': 1e-4,
    'm2':-4,
    'lam':5.276,
    'epochs': 80000,
'model_save_path': 'normalizing_flow_best_model_12_80000.pt',
    'loss_save_path':'normalizing_flow_loss_12_80000}.npy'
},{
    'L': 14,
    'batch_size': 1024,
    'lr': 1e-4,
    'm2':-4,
    'lam':5.113,
    'epochs': 100000,
    'model_save_path': 'normalizing_flow_best_model_14_100000.pt',
    'loss_save_path':'normalizing_flow_loss_14_100000}.npy'
},]

def create_mask(L):
    indices = torch.arange(L)
    mask = (indices[:,None] + indices[None,:]) % 2 == 0
    return mask.flatten() # 返回 (L*L,) 的布尔张量

def compute_action(phi, config):
    # phi shape: [batchsize, L, L]
    phi_up = torch.roll(phi, shifts=-1, dims=1)
    phi_down = torch.roll(phi, shifts=1, dims=1)
    phi_right = torch.roll(phi, shifts=1, dims=2)
    phi_left = torch.roll(phi, shifts=-1, dims=2)

    # 动能项 (离散拉普拉斯算子部分)
    kinetic_term = 4 * phi * phi - phi * (phi_right + phi_left + phi_up + phi_down)
    potential_term = config['m2'] * phi * phi + config['lam'] * phi ** 4

    # 对整个 grid 求和，得到标量 Action
    action = (kinetic_term + potential_term).sum(dim=(1, 2))
    return action

class StNet(nn.Module):
    def __init__(self,input_dim,hide_dim):
        super(StNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hide_dim),
            nn.LeakyReLU(),
            nn.Linear(hide_dim, hide_dim),
            nn.LeakyReLU(),
            nn.Linear(hide_dim, input_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class RealNVP(nn.Module):
    def __init__(self,coupling_layer_num=8, hidden_dim=128, config=None):
        super(RealNVP, self).__init__()
        self.config = config
        self.coupling_layer_num = coupling_layer_num
        input_dim = config['L']*config['L'] // 2
        self.register_buffer('base_mask', create_mask(config['L']))
        self.sNet = nn.ModuleList([StNet(input_dim, hidden_dim) for _ in range(coupling_layer_num)])
        self.tNet = nn.ModuleList([StNet(input_dim, hidden_dim) for _ in range(coupling_layer_num)])

    def forward(self, phi):
        # phi 形状: (batch, L, L) -> (batch, L*L)
        z = phi.view(phi.size(0), -1)
        log_r_z = torch.sum(-0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi), dim=1)
        log_det_jacobian = 0
        for i in range(self.coupling_layer_num):
            mask = self.base_mask if i % 2 == 0 else ~self.base_mask
            z_a = z[:,mask]
            z_b = z[:,~mask]
            s_out = self.sNet[i](z_b)
            s_out = torch.tanh(s_out)
            t_out = self.tNet[i](z_b)
            z_a = z_a * torch.exp(s_out) + t_out
            log_det_jacobian += torch.sum(s_out,dim=1)
            # 重新组装
            z_new = torch.empty_like(z)
            z_new[:, mask] = z_a
            z_new[:, ~mask] = z_b
            z = z_new
        # 还原形状
        phi_out = z.view(z.size(0), self.config['L'], self.config['L'])
        log_q = log_r_z - log_det_jacobian
        return phi_out, log_q

    def compute_loss(self, phi, log_q):
        loss = torch.mean(log_q, dim=0) + torch.mean(compute_action(phi, self.config))
        return loss


def train(config):

    model = RealNVP(config = config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    Loss_history = []
    model_save_path = config['model_save_path']
    loss_save_path = config['loss_save_path']
    best_loss = float('inf')

    model.train()
    for epoch in range(1,config['epochs']+1):
        optimizer.zero_grad()
        z = torch.randn(config['batch_size'], config['L'], config['L'], device=device)
        phi, log_det_jacobian = model(z)
        loss = model.compute_loss(phi, log_det_jacobian)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        Loss_history.append(loss_val)

        # 打印日志
        if epoch % 100 == 0:
            print(f"迭代 {epoch:6d}/{config['epochs']} | Loss: {loss.item():.4f}")
            # 保存最新模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, f"normalizing_flow_latest_checkpoint_{config['L']}_{config['epochs']}.pt")

        # 如果 Loss 创新低，保存最优模型
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), model_save_path)
    np.save(loss_save_path, np.array(Loss_history))
    print(f"{config['L']} model training finished！")

if __name__ == "__main__":
    for _ in range(len(CONFIG)):
        train(CONFIG[_])