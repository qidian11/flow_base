import torch
import numpy as np
import os
# 直接导入你的训练脚本组件，完美复用 CONFIG 自带路径
from final_normalizing import CONFIG, FlowModel, FreeFieldPrior, compute_action, device


def generate_and_save_configs(N=1000000, batch_size=2500):
    print(f"🔥 当前运行设备: {device}")

    # 1. 初始化模型与先验
    model = FlowModel(CONFIG).to(device)
    # 严格使用你推导出的最优先验质量参数
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=0.6005269985).to(device)

    dtype_tensor = torch.float64 if CONFIG.get('double_precision', False) else torch.float32
    if CONFIG.get('double_precision', False):
        model = model.double()
        prior = prior.double()

    # 2. 加载最佳检查点
    checkpoint_path = f"best_{CONFIG['base_name']}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ 找不到最优模型权重: {checkpoint_path}")

    print(f"📥 正在加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(clean_dict)
    model.eval()

    # 3. 批量生成提案 (Proposals)
    print(f"🚀 开始生成 {N} 个构型 (Batch Size: {batch_size})...")
    L = CONFIG['L']
    all_phis = torch.empty((N, 1, L, L), dtype=dtype_tensor, device=device)
    all_s = torch.empty(N, dtype=dtype_tensor, device=device)
    all_log_qs = torch.empty(N, dtype=dtype_tensor, device=device)

    dummy_progress = torch.tensor(1.0, device=device, dtype=dtype_tensor)
    enforce_sym = CONFIG.get('enforce_z2_sym', False)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            curr_batch = min(batch_size, N - i)
            z, log_p_z = prior.sample(curr_batch)
            phi, log_det_J = model(z, dummy_progress, enforce_sym=enforce_sym)

            all_phis[i:i + curr_batch] = phi
            all_s[i:i + curr_batch] = compute_action(phi)
            all_log_qs[i:i + curr_batch] = log_p_z - log_det_J

            if (i + curr_batch) % 100000 == 0:
                print(f"   已生成 {i + curr_batch} / {N} 个提案...")

    # 4. 执行 CPU 端 MCMC 接受/拒绝链
    print("🔄 正在执行 MCMC 接受/拒绝逻辑 (构造马尔可夫链)...")
    s_np = all_s.cpu().numpy()
    log_q_np = all_log_qs.cpu().numpy()
    log_rands_np = np.log(np.random.rand(N))

    accepted_indices = np.zeros(N, dtype=int)
    accepted_count = 0

    curr_s_val = s_np[0]
    curr_log_q_val = log_q_np[0]
    accepted_indices[0] = 0

    for i in range(1, N):
        prop_s_val = s_np[i]
        prop_log_q_val = log_q_np[i]
        log_acc_ratio = (-prop_s_val - prop_log_q_val) - (-curr_s_val - curr_log_q_val)

        if log_rands_np[i] < log_acc_ratio:
            curr_s_val = prop_s_val
            curr_log_q_val = prop_log_q_val
            accepted_indices[i] = i
            accepted_count += 1
        else:
            accepted_indices[i] = accepted_indices[i - 1]

    print(f"✅ 生成完毕！ML MCMC 接受率: {accepted_count / (N - 1):.2%}")

    # 5. 提取构型并保存 (🌟 直接使用你的自带配置路径)
    idx_tensor = torch.tensor(accepted_indices, device=device, dtype=torch.long)
    chain_phis = all_phis[idx_tensor].cpu().numpy()

    save_path = CONFIG['phi_ensemble_save_path']
    np.savez_compressed(save_path, configs=chain_phis)
    print(f"💾 构型已成功保存至: {save_path}")


if __name__ == '__main__':
    generate_and_save_configs(N=1000000)