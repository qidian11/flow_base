import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import glob


print(torch.cuda.is_available())  # 如果输出 True，说明能找到显卡
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0)) # 这会直接打印出你租用的显卡型号，比如 "NVIDIA GeForce RTX 4090"
# 优先检测 CUDA (AutoDL环境)，其次检测 XPU (本地 B580 环境)，最后回退到 CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
)
print(f"运行设备: {device}")


def is_valid_checkpoint(filepath):
    """
    轻量级测试 checkpoint 文件是否完整无损。
    只读取元数据到 CPU，不加载权重到显存，速度极快。
    """
    try:
        torch.load(filepath, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def auto_find_latest_checkpoint(config):
    """
    自动搜索当前目录下架构匹配的断点文件。
    拥有自动跳过损坏文件，并执行降级回退的机制。
    搜索优先级：健康的 latest > 健康的 best > 健康的 里程碑
    """
    # 提取通用的核心匹配字符串 (忽略双精度开关和迭代次数的差异)
    base_pattern = (
        f"prior_cnn_res_model_double_precision_*_"
        f"{config['L']}_coupling_layers_{config['coupling_layers']}_"
        f"hidden_layers_{config['hidden_layers']}_hidden_channels_{config['hidden_channels']}_"
        f"iterations_*.pt"
    )

    # 辅助函数：根据前缀获取按修改时间从新到老排序的文件列表
    def get_sorted_files(prefix):
        files = glob.glob(prefix + base_pattern)
        return sorted(files, key=os.path.getmtime, reverse=True)

    # ================= 1. 优先尝试找 latest =================
    for file in get_sorted_files("latest_"):
        if is_valid_checkpoint(file):
            return file
        else:
            print(f"⚠️ 警告: 检测到损坏的 latest 断点并已自动跳过 -> {file}")

    # ================= 2. 找不到/全坏了，退而求其次找 best =================
    for file in get_sorted_files("best_"):
        if is_valid_checkpoint(file):
            print(f"🔄 未找到可用的 latest，已自动回退到最新的 best 断点 -> {file}")
            return file
        else:
            print(f"⚠️ 警告: 检测到损坏的 best 断点并已自动跳过 -> {file}")

    # ================= 3. 终极保底：找你每5000步存的里程碑 =================
    # 里程碑文件没有 latest_ 或 best_ 前缀
    for file in get_sorted_files(""):
        filename = os.path.basename(file)
        # 排除掉已经被 glob 匹配到的 latest_ 和 best_ 文件
        if not filename.startswith(("latest_", "best_")):
            if is_valid_checkpoint(file):
                print(f"🔄 best 也不可用，已自动回退到里程碑断点 -> {file}")
                return file

    # 4. 彻底弹尽粮绝，只能从头开始
    return None


# ==========================================
# 1. 物理参数配置
# ==========================================
# CONFIG = {
#     'L': 14,  # 晶格大小 (对应实验 E5)
#     'm_sq': -4.0,  # m^2 (质量的平方，破缺相)
#     'lam': 5.113,  # lambda (耦合常数)
#     'batch_size': 2048,  # 批大小
#     'lr': 1e-6,  # 学习率
#     'use_scheduler': False, # <--- 新增开关，方便以后随时切回退火
#     'iterations': 45000,  # 训练迭代次数
#     'coupling_layers': 32,
#     'kernel_size': 3,
#     'hidden_layers': 6,  # 实际上是6*3
#     'hidden_channels': 16,
#     'double precision': True,
# }
CONFIG = {
    'L': 14,  # 晶格大小 (对应实验 E5)
    'm_sq': -4.0,  # m^2 (质量的平方，破缺相)
    'lam': 5.113,  # lambda (耦合常数)
    'batch_size': 1024,  # 批大小
    'lr': 1e-5,  # 学习率
    'use_scheduler': False, # <--- 新增开关，方便以后随时切回退火
    'iterations': 25000,  # 训练迭代次数
    'coupling_layers': 14,
    'kernel_size': 6,
    'hidden_layers': 6,  # 实际上是6*3
    'hidden_channels': 16,
    'double precision': False,
}
save_path = f"best_prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_kernel_size_{CONFIG['kernel_size']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
loss_save_path = f"prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_loss_history_coupling_layers_{CONFIG['coupling_layers']}_kernel_size_{CONFIG['kernel_size']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npy"
checkpoint_path = f"latest_prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_kernel_size_{CONFIG['kernel_size']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.pt"
phi_ensemble_save_path = f"phi_ensemble_prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_loss_history_coupling_layers_{CONFIG['coupling_layers']}_kernel_size_{CONFIG['kernel_size']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}.npz"
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
    """
    phi_up = torch.roll(phi, shifts=-1, dims=2)
    phi_down = torch.roll(phi, shifts=1, dims=2)
    phi_left = torch.roll(phi, shifts=-1, dims=3)
    phi_right = torch.roll(phi, shifts=1, dims=3)

    laplacian = 4 * phi - phi_up - phi_down - phi_left - phi_right
    action_density = phi * laplacian + CONFIG['m_sq'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)

    return torch.sum(action_density, dim=(1, 2, 3))


# ==========================================
# 3. 掩码生成与卷积上下文网络 (保持原样)
# ==========================================
def create_checkerboard_mask(L):
    indices = torch.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask_2d.view(1, 1, L, L).float()


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, padding_mode='circular')
        self.act1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, padding_mode='circular')
        self.act2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, padding_mode='circular')
        self.act3 = nn.LeakyReLU(0.01)

    def forward(self, x):
        return x + self.act3(self.conv3(self.act2(self.conv2(self.act1(self.conv1(x))))))


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4,kernel_size=3):
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
# 4. 【新增】自由场先验 (Free Field Prior)
# ==========================================
class FreeFieldPrior(nn.Module):
    def __init__(self, L, m_sq_prior):
        super().__init__()
        self.L = L
        self.V = L * L

        # 构建动量空间的晶格拉普拉斯算子本征值 K(p)
        p = torch.arange(L) * 2.0 * math.pi / L
        P1, P2 = torch.meshgrid(p, p, indexing='ij')

        # 自由场作用量算符: K(p) = m_0^2 + 4 * sin^2(p_1/2) + 4 * sin^2(p_2/2)
        K = m_sq_prior + 4.0 * torch.sin(P1 / 2.0) ** 2 + 4.0 * torch.sin(P2 / 2.0) ** 2

        # 使用 register_buffer 使得它们能够自动随模型 .to(device) 和 .double() 切换
        self.register_buffer('sqrt_2K', torch.sqrt(2.0 * K).view(1, 1, L, L))

        # 预计算常数项用于 log p(phi)
        # log_det_factor = 0.5 * sum_p log(2K(p))
        log_det_factor = 0.5 * torch.sum(torch.log(2.0 * K))
        self.register_buffer('log_det_factor', log_det_factor)

        # V/2 * log(2pi)
        const_factor = 0.5 * self.V * math.log(2.0 * math.pi)
        self.register_buffer('const_factor', torch.tensor(const_factor))

    def sample(self, batch_size):
        """
        通过在动量空间滤波生成完美的自由场配置，并返回其精确对数概率。
        """
        # 1. 在实空间生成完全独立的标准高斯白噪声
        eta = torch.randn(batch_size, 1, self.L, self.L, device=self.sqrt_2K.device, dtype=self.sqrt_2K.dtype)

        # 2. 转换到动量空间 (norm='ortho' 保证酉变换，雅可比为1)
        eta_k = torch.fft.fftn(eta, dim=(-2, -1), norm="ortho")

        # 3. 注入物理相关性 (用 1/sqrt(2K) 缩放振幅)
        phi_k = eta_k / self.sqrt_2K

        # 4. 逆变换回实空间得到场配置 (直接取实部即可)
        phi_free = torch.fft.ifftn(phi_k, dim=(-2, -1), norm="ortho").real

        # 5. 精确计算这个场的先验生成概率: log p(phi) = log p(eta) - log|det J|
        # 由于我们保留了原始白噪声 eta，直接用 eta 计算最快、最数值稳定！
        log_p_eta = -0.5 * torch.sum(eta ** 2, dim=(1, 2, 3)) - self.const_factor
        log_p_phi = log_p_eta + self.log_det_factor
        return phi_free, log_p_phi


# ==========================================
# 5. 修改：流模型 (仅负责非线性双射，不再自带标准正态噪声)
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, L, coupling_layers=12, hidden_channels=16, num_hidden_layers=12, kernel_size=3):
        super().__init__()
        self.L = L
        self.coupling_layers = coupling_layers
        self.register_buffer('base_mask', create_checkerboard_mask(L))
        self.context_nets = nn.ModuleList(
            [ConvContextNet(hidden_channels=hidden_channels,
                            num_hidden_layers=num_hidden_layers,
                            kernel_size=kernel_size)
             for _ in range(coupling_layers)])
        print(f'当前模型耦合层：{coupling_layers} 通道数：{hidden_channels}， 隐藏层：{num_hidden_layers}, 卷积核：{kernel_size}x{kernel_size}')

    def forward(self, z):
        """
        接收来自 FreeFieldPrior 的自由场 z，将其映射为相互作用场 phi
        仅返回变形后的 phi 和流模型产生的对数雅可比矩阵
        """
        phi = z
        log_det_jacobian = 0

        for i in range(self.coupling_layers):
            current_mask = self.base_mask if i % 2 == 0 else (1.0 - self.base_mask)
            phi_frozen = current_mask * phi

            st_out = self.context_nets[i](phi_frozen)
            # s_out = torch.tanh(st_out[:, 0:1, :, :])
            s_out = st_out[:, 0:1, :, :]
            t_out = st_out[:, 1:2, :, :]

            update_mask = 1.0 - current_mask
            phi_updated = update_mask * (phi * torch.exp(s_out) + t_out)
            phi = phi_frozen + phi_updated

            log_det_jacobian += torch.sum(update_mask * s_out, dim=(1, 2, 3))

        return phi, log_det_jacobian


# ==========================================
# 6. 自训练循环
# ==========================================
def train(save_path=CONFIG['save_path'], loss_save_path=CONFIG['loss_save_path'], resume=True,
          checkpoint_path=CONFIG['checkpoint_path']):
    L = CONFIG['L']

    # 实例化流模型
    model = FlowModel(L=L, coupling_layers=CONFIG['coupling_layers'],
                      hidden_channels=CONFIG['hidden_channels'],
                      num_hidden_layers=CONFIG['hidden_layers'],
                      kernel_size=CONFIG['kernel_size']).to(device)

    # 【新增】实例化自由场先验 (因为处于破缺相，使用质量的绝对值作为先验的正质量 m_0^2)
    prior_m_sq = abs(CONFIG['m_sq'])
    prior = FreeFieldPrior(L=L, m_sq_prior=prior_m_sq).to(device)

    if CONFIG['double precision']:
        model = model.double()
        prior = prior.double()

    # ==========================================
    # 【新增代码区】开启 PyTorch 2.0 编译加速
    # ==========================================
    # 确保只在 AutoDL 的 CUDA 环境下开启（本地 XPU 暂不支持或不需要）
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("🚀 正在开启 torch.compile() 加速...")
        # reduce-overhead 模式专门用来对付你这种 "单次计算量小、步骤多" 的开销瓶颈
        model = torch.compile(model)
    # ==========================================

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    # 开启退火
    scheduler = None
    if CONFIG.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['iterations'], eta_min=1e-5
        )

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
                        if isinstance(v, torch.Tensor):
                            state[k] = v.double()

            for param_group in optimizer.param_groups:
                param_group['lr'] = CONFIG['lr']

            start_iteration = checkpoint['iteration'] + 1

            if 'history_loss' in checkpoint:
                history_loss = checkpoint['history_loss']
                print(f"成功从 checkpoint 恢复历史 Loss，当前有 {len(history_loss)} 条未平滑原始数据。")

            print(f"恢复成功！将从第 {start_iteration} 步继续训练至 {CONFIG['iterations']} 步。")
        else:
            print("未找到断点文件，将从头开始训练。")

    model.train()
    print("开始基于自由场先验的自训练...")

    for iteration in range(start_iteration, CONFIG['iterations'] + 1):
        optimizer.zero_grad()

        # 1. 从自由场先验中取样，得到自由场构型 z 及其确切对数概率 log_p_z
        z, log_p_z = prior.sample(CONFIG['batch_size'])

        # 2. 通过流模型，加入 phi^4 相互作用的微调变形
        phi, log_det_J = model(z)

        # 3. 计算最终的生成概率密度: log q(phi) = log_prior(z) - log|det(d_phi / d_z)|
        log_q = log_p_z - log_det_J

        # 4. 计算目标作用量 S(phi)
        S_phi = compute_action(phi)

        # 5. 最小化 Shifted KL 散度
        loss = torch.mean(log_q + S_phi)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        history_loss.append(loss_val)

        if iteration % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"迭代 {iteration:6d}/{CONFIG['iterations']} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': loss_val,
                'history_loss': history_loss,
            }, checkpoint_path)

        if loss_val < best_loss and iteration >= 10000:
            best_loss = loss_val
            print(f"🌟 第 {iteration}步发现更优模型！当前最佳 Loss: {best_loss:.4f}，正在更新 best 断点...")
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': best_loss,
                'history_loss': history_loss,
            }, save_path)

        if iteration % 1000 == 0:
            np.save(loss_save_path, np.array(history_loss))

            # ==========================================
            # 【新增代码】每 5000 步保存一次当前迭代次数的模型
            # ==========================================
            if iteration % 5000 == 0:
                # 注意：这里的变量是当前的 iteration，而不是 CONFIG['iterations']
                milestone_path = f"prior_cnn_res_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{iteration}.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': loss_val,
                    'history_loss': history_loss,
                }, milestone_path)
                print(f"📦 里程碑达成！已保存第 {iteration} 步的完整独立断点: {milestone_path}")
            # ==========================================

    print("训练结束！")


if __name__ == "__main__":
    if CONFIG.get('double precision', False):
        torch.set_default_dtype(torch.float64)
        print('开启双精度')
    train(CONFIG['save_path'], CONFIG['loss_save_path'])