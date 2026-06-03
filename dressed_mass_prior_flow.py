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
    # 🌟 新增：提取分支深度参数
    depth_str = f"depth_{config.get('branch_depth', 3)}"

    # 提取通用的核心匹配字符串 (忽略双精度开关和迭代次数的差异)
    base_pattern = (
        f"dressed_mass_prior_model_double_precision_*_"
        f"{config['L']}_coupling_layers_{config['coupling_layers']}_"
        f"kernel_size_{config['kernel_size']}_{depth_str}_"  # 🌟 插入 depth_str
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
# CONFIG = {
#     'Type': 'Prior_CNN',
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
    'lr': 1e-3,  # 学习率
    'use_scheduler': True, # <--- 新增开关，方便以后随时切回退火
    'iterations': 25000,  # 训练迭代次数
    'coupling_layers': 12,
    'kernel_size': 3,
    'hidden_layers': 4,  # 实际上是6*3
    'branch_depth': 2,  # 🌟 新增：ResBlock内部的卷积层数
    'hidden_channels': 16,
    'double precision': False,
}
# 🌟 新增：拼接带有 depth 的公用后缀
depth_str = f"depth_{CONFIG.get('branch_depth', 3)}"
base_suffix = f"dressed_mass_prior_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_kernel_size_{CONFIG['kernel_size']}_{depth_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{CONFIG['iterations']}"

save_path = f"best_{base_suffix}.pt"
loss_save_path = f"{base_suffix}_loss_history.npy"
checkpoint_path = f"latest_{base_suffix}.pt"
phi_ensemble_save_path = f"phi_ensemble_{base_suffix}.npz"

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
    def __init__(self, channels, kernel_size=3, branch_depth=3):
        super().__init__()
        # 动态计算 padding
        pad = kernel_size // 2

        # 🌟 修改：使用动态网络容器堆叠指定数量的 卷积+激活层
        layers = []
        for _ in range(branch_depth):
            layers.append(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=pad, padding_mode='circular'))
            layers.append(nn.LeakyReLU(0.01))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # 🌟 现在的残差连接极其简洁
        return x + self.block(x)


class ConvContextNet(nn.Module):
    def __init__(self, hidden_channels=8, num_hidden_layers=4,kernel_size=3, branch_depth=3):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=1, stride=1, padding=0, padding_mode='circular'))
        layers.append(nn.LeakyReLU(0.01))

        for _ in range(num_hidden_layers):
            layers.append(ResBlock(hidden_channels,kernel_size=kernel_size,branch_depth=branch_depth))
        # 修复2：动态计算最后一层的 padding
        pad = kernel_size // 2
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=kernel_size, stride=1, padding=pad, padding_mode='circular'))
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
# [新增] 泛函极值求解器：计算理论最优的穿衣质量 (Dressed Mass)
# ==========================================
def solve_gap_equation_lattice(L, m_sq_bare, lam, device, tol=1e-7, max_iter=1000):
    """
    通过不动点迭代严格求解格点上的自洽能隙方程，
    寻找变分最优的高斯协方差核参数。
    """
    dtype = torch.float64  # 极值求解必须使用高精度
    p = torch.arange(L, device=device, dtype=dtype) * 2.0 * math.pi / L
    P1, P2 = torch.meshgrid(p, p, indexing='ij')

    # 动量空间中的格点拉普拉斯本征值 K_hat
    K_hat = 4.0 * torch.sin(P1 / 2.0) ** 2 + 4.0 * torch.sin(P2 / 2.0) ** 2

    # 初始猜测：为防止破缺相裸质量为负导致对数奇点，取绝对值作为安全起点
    m_eff_sq = torch.tensor(abs(m_sq_bare), device=device, dtype=dtype)

    for i in range(max_iter):
        # 计算单圈蝌蚪图积分 (Tadpole Integral): (1/V) * Sum_p [1 / (p^2 + m_eff^2)]
        tadpole = torch.mean(1.0 / (K_hat + m_eff_sq))

        # 严格代入波戈留波夫变分导出的极值方程
        # 注意：此处必须使用物理真实的 m_sq_bare (包含 m^2 = -4.0 的负号)
        m_new_sq = m_sq_bare + (lam / 2.0) * tadpole

        # 数值保护：若迭代跌入绝对不稳定的纯破缺死区，施加极小的正截断
        if m_new_sq <= 0:
            m_new_sq = torch.tensor(1e-4, device=device, dtype=dtype)

        # 检查收敛
        if torch.abs(m_new_sq - m_eff_sq) < tol:
            print(f"🌟 最优高斯核求解成功! (迭代 {i} 次) | 裸质量^2: {m_sq_bare} -> 穿衣质量^2: {m_new_sq.item():.4f}")
            return m_new_sq.item()

        m_eff_sq = m_new_sq

    print(f"⚠️ 能隙方程达到最大迭代次数未完全收敛，返回最后近似值: {m_eff_sq.item():.4f}")
    return m_eff_sq.item()


# ==========================================
# 5. 修改：流模型 (仅负责非线性双射，不再自带标准正态噪声)
# ==========================================
class FlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config['L']
        self.coupling_layers = config['coupling_layers']
        self.register_buffer('base_mask', create_checkerboard_mask(self.L))

        # 从 config 优雅解包
        self.context_nets = nn.ModuleList([
            ConvContextNet(
                hidden_channels=config['hidden_channels'],
                num_hidden_layers=config['hidden_layers'],
                kernel_size=config.get('kernel_size', 3),
                branch_depth=config.get('branch_depth', 3)
            ) for _ in range(self.coupling_layers)
        ])
        print(
            f"当前单核模型耦合层：{self.coupling_layers} 通道数：{config['hidden_channels']}， 隐藏层：{config['hidden_layers']}, 卷积核：{config.get('kernel_size', 3)}")


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
            # 🌟 统一仿射公式为逆向方程： y = (x - t) * exp(-s)
            phi = phi_frozen + update_mask * ((phi - t_out) * torch.exp(-s_out))
            log_det_jacobian += torch.sum(update_mask * (-s_out), dim=(1, 2, 3))

        return phi, log_det_jacobian


# ==========================================
# 6. 自训练循环
# ==========================================
def train(config, resume = True):
    save_path = config['save_path']
    loss_save_path = config['loss_save_path']
    checkpoint_path = config['checkpoint_path']
    L = config['L']

    # 实例化流模型
    model = FlowModel(config).to(device)

    # ====== [核心修改] 注入变分最优先验 ======
    print("\n" + "=" * 70)
    print("🛠️ 正在求解变分泛函极值，计算最优先验 (Dressed Gaussian Prior)...")
    m_sq_dressed = solve_gap_equation_lattice(CONFIG['L'], CONFIG['m_sq'], CONFIG['lam'], device)
    print("=" * 70 + "\n")

    # 将原先的粗糙估计 abs(CONFIG['m_sq']) 替换为理论极值 m_sq_dressed
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=m_sq_dressed).to(device)

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

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
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

            print(f"恢复成功！将从第 {start_iteration} 步继续训练至 {config['iterations']} 步。")
        else:
            print("未找到断点文件，将从头开始训练。")

    model.train()
    print("开始基于自由场先验的自训练...")

    for iteration in range(start_iteration, config['iterations'] + 1):
        optimizer.zero_grad()

        # 1. 从自由场先验中取样，得到自由场构型 z 及其确切对数概率 log_p_z
        z, log_p_z = prior.sample(config['batch_size'])

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
                # 🌟 加入 branch_depth 支持，保持命名系统一致
                depth_str = f"depth_{CONFIG.get('branch_depth', 3)}"
                milestone_path = f"dressed_mass_prior_model_double_precision_{CONFIG['double precision']}_{CONFIG['L']}_coupling_layers_{CONFIG['coupling_layers']}_kernel_size_{CONFIG['kernel_size']}_{depth_str}_hidden_layers_{CONFIG['hidden_layers']}_hidden_channels_{CONFIG['hidden_channels']}_iterations_{iteration}.pt"

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
    train(CONFIG)