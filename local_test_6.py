import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

# --- 配置 ---
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'delta': 0.5,
    'thermal_steps': 2000,  # 因为并行了，热化步数可以适当减少或保持
    'n_samples_needed': 1000000,  # 总共需要的样本数
    'batch_size': 4096,  # CPU内存允许的情况下尽可能大！比如 1024-8192
    'bin_size': 100,
    'bootstrap_time': 200
}

# 优化3: 使用 Float32 (CPU上速度快一倍)
DTYPE = torch.float32

# 定义卷积核 (用于计算邻居和)
# 形状: [Out_channels, In_channels, H, W] -> [1, 1, 3, 3]
KERNEL = torch.tensor([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=DTYPE).reshape(1, 1, 3, 3)


def get_neighbor_sum_conv(phi):
    """
    使用卷积计算邻居和 (CPU优化版)
    输入: [Batch, L, L]
    输出: [Batch, L, L]
    """
    # 1. 增加 Channel 维度: [B, 1, L, L]
    x = phi.unsqueeze(1)
    # 2. 循环填充 (处理周期性边界): [B, 1, L+2, L+2]
    #    Padding顺序: (Left, Right, Top, Bottom)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='circular')
    # 3. 卷积
    out = F.conv2d(x_pad, KERNEL)
    # 4. 去掉 Channel 维度
    return out.squeeze(1)


# 优化4: 编译函数 (减少 Python Overhead)
# @torch.compile
def metropolis_batch_step(phi, mask_red, mask_black):
    """
    并行化的红黑 Metropolis 更新
    phi shape: [Batch, L, L]
    """
    L = CONFIG['L']
    m2 = CONFIG['m2']
    lam = CONFIG['lam']
    delta = CONFIG['delta']
    term1 = (4 + m2)

    # 辅助函数：计算 Action 变化并更新
    def update_phase(current_phi, mask):
        # 1. 计算邻居和 (利用卷积)
        # 注意：这里虽然对全图做了卷积，但由于 Batch 很大，卷积算子的并行度极高，
        # 通常比手动切片再拼凑索引要快，且代码更利用缓存。
        neighbor_sum = get_neighbor_sum_conv(current_phi)

        # 2. 生成随机扰动 [Batch, L, L]
        change = (torch.rand_like(current_phi) - 0.5) * 2 * delta
        phi_proposal = current_phi + change

        # 3. 计算 Delta S (向量化计算)
        # S = (4+m2)*phi^2 + lam*phi^4 - phi*neighbor
        S_old = term1 * current_phi ** 2 + lam * current_phi ** 4 - current_phi * neighbor_sum
        S_new = term1 * phi_proposal ** 2 + lam * phi_proposal ** 4 - phi_proposal * neighbor_sum
        delta_S = S_new - S_old

        # 4. Metropolis 判据
        accept_prob = torch.rand_like(current_phi)
        accept_condition = (delta_S < 0) | (torch.exp(-delta_S) > accept_prob)

        # 5. 只更新 Mask 对应的点
        # mask & accept_condition
        update_mask = mask & accept_condition

        # 6. 应用更新
        new_phi = torch.where(update_mask, phi_proposal, current_phi)

        # 统计接受率 (对 Batch 和 Space 平均)
        acc_rate = update_mask.float().mean()
        return new_phi, acc_rate

    # === Phase 1: Red ===
    phi, acc_red = update_phase(phi, mask_red)

    # === Phase 2: Black ===
    phi, acc_black = update_phase(phi, mask_black)

    return phi, (acc_red + acc_black) / 2


# ... (G_t 计算函数保持不变，注意加上 batch 维度的处理) ...
def calculate_G_t_batch(ensemble_batch):
    # ensemble_batch: [Total_Samples, L, L]
    # 由于数据量太大，这里建议分块处理或者简化计算
    # 这里为了演示，直接沿用原来的逻辑，但要注意内存

    # 简单起见，直接用原来的逻辑，PyTorch会自动广播
    # 为了避免内存爆炸，建议把 [Total_Samples] 维度的平均放在循环里做

    L = CONFIG['L']
    G_t_list = []

    # 预先计算全格子的平均磁化强度 <phi>
    # shape: [Total_Samples]
    expected_phi = ensemble_batch.mean(dim=(1, 2))

    for t in range(L):
        G_t_l_list = []
        for l in range(L):
            # Roll data
            shifted = torch.roll(ensemble_batch, shifts=(t, l), dims=(1, 2))

            # Correlation: <phi(0) * phi(t, l)>
            # shape: [Total_Samples]
            corr = (ensemble_batch * shifted).mean(dim=(1, 2))

            # Connected: <phi*phi> - <phi><phi>
            # 注意这里要减去每个样本自己的 <phi>，而不是所有样本的平均
            conn = corr - expected_phi * expected_phi  # 近似，严格来说应该是 <phi>_shifted

            G_t_l_list.append(conn.mean())  # 对样本求平均

        G_t_list.append(torch.stack(G_t_l_list).mean())

    return torch.stack(G_t_list)


def main():
    torch.set_num_threads(8)  # 根据你的 CPU 核心数调整，例如 8 或 16
    print(f"Running on CPU with Batch Size: {CONFIG['batch_size']}")

    L = CONFIG['L']
    BS = CONFIG['batch_size']

    # 1. 初始化 [Batch, L, L]
    phi = torch.ones(BS, L, L, dtype=DTYPE) * 0.6

    # 准备 Mask (广播到 Batch 维度)
    coords = torch.arange(L)
    i, j = torch.meshgrid(coords, coords, indexing='ij')
    mask_red = ((i + j) % 2 == 0).unsqueeze(0).expand(BS, L, L)
    mask_black = ((i + j) % 2 == 1).unsqueeze(0).expand(BS, L, L)

    # 2. Thermalization
    print("Start Thermalization...")
    t0 = time.time()
    for step in range(CONFIG['thermal_steps']):
        phi, acc = metropolis_batch_step(phi, mask_red, mask_black)
        if step % 100 == 0:
            print(f"Step {step}: Acc {acc:.3f}")
    print(f"Thermalization done in {time.time() - t0:.2f}s")

    # 3. Sampling
    # 计算需要多少个 Batch 步数才能凑够总样本
    # 比如总共要 1,000,000 个样本，Batch=4096，那只需要跑 ~245 步
    n_batches = CONFIG['n_samples_needed'] // BS
    if n_batches == 0: n_batches = 1

    print(f"Start Sampling... Need {n_batches} batches to get {CONFIG['n_samples_needed']} samples.")

    ensemble_list = []
    t0 = time.time()

    for step in range(n_batches):
        # 每一“步”其实产生了 Batch_Size 个样本
        # 为了去自相关，通常在两个采样点之间多跑几步 (Decorrelation steps)
        for _ in range(10):  # 间隔 10 步取一次样
            phi, acc = metropolis_batch_step(phi, mask_red, mask_black)

        # 收集样本: 将显存/内存中的 Tensor 复制出来
        # clone() 很重要
        ensemble_list.append(phi.clone())

        if step % 10 == 0:
            print(f"Sampling Batch {step}/{n_batches}")

    # 4. 数据处理
    # 拼接所有 Batch: [N_batches * Batch_Size, L, L]
    ensemble_tensor = torch.cat(ensemble_list, dim=0)
    print(f"Total Samples Collected: {ensemble_tensor.shape[0]}")
    print(f"Sampling done in {time.time() - t0:.2f}s")

    # 计算 G(t) (使用简化的函数)
    # 注意：如果 ensemble_tensor 太大，这里可能会内存溢出，需要分批计算
    G_t = calculate_G_t_batch(ensemble_tensor)

    # Effective Mass
    G_plus = torch.roll(G_t, shifts=-1, dims=0)
    G_minus = torch.roll(G_t, shifts=1, dims=0)
    cosh_m = (G_plus + G_minus) / (2 * G_t + 1e-8)
    # 简单的 clamp 防止 nan
    m_eff = torch.acosh(torch.clamp(cosh_m, min=1.00001))

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(G_t.numpy(), 'o-', label='G(t)')
    axs[0].set_yscale('log')
    axs[0].set_title('Green Function')

    axs[1].plot(m_eff.numpy()[1:-1], 'o-', color='red', label='Mass')
    axs[1].set_title('Effective Mass')
    axs[1].set_ylim(0, 2)  # 根据物理预期调整
    plt.show()


if __name__ == '__main__':
    main()