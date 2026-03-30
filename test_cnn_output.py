import torch
import numpy as np
from cnn_flow import FlowModel, compute_action, CONFIG, device  # 复用你之前的定义


def load_trained_model(checkpoint_path, L):
    """加载保存的模型权重"""
    model = FlowModel(L=L).to(device)
    # map_location 确保在没有 GPU 的机器上也能加载
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 如果你保存的是整个 dict (包含 optimizer)，则取 model_state_dict
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()  # 必须切换到评估模式
    return model


def produce_ensemble(model, total_n=10000, batch_size=1024):
    """
    批量生成物理构型集成
    1. 并行生成 Proposal
    2. 串行构建 Markov Chain (MH 验证)
    """
    all_phis = []
    all_log_qs = []

    # --- 第一步：并行采样 (GPU 加速) ---
    print(f"正在并行生成 {total_n} 个提案...")
    with torch.no_grad():
        for _ in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - len(all_phis))
            z = torch.randn(current_batch, 1, CONFIG['L'], CONFIG['L'], device=device)
            phi, log_q = model(z)
            all_phis.append(phi)
            all_log_qs.append(log_q)

    phi_proposals = torch.cat(all_phis, dim=0)
    log_q_proposals = torch.cat(all_log_qs, dim=0)
    # 提前计算所有提案的 Action S(phi)
    s_proposals = compute_action(phi_proposals)

    # --- 第二步：串行 MH 验证 (构建马尔可夫链) ---
    print("正在通过 MH 算法构建马尔可夫链...")
    ensemble = []
    accepted_count = 0

    # 初始化链的第一个状态
    curr_phi = phi_proposals[0:1]
    curr_log_q = log_q_proposals[0:1]
    curr_s = s_proposals[0:1]

    for i in range(1, total_n):
        prop_phi = phi_proposals[i:i + 1]
        prop_log_q = log_q_proposals[i:i + 1]
        prop_s = s_proposals[i:i + 1]

        # 核心公式[cite: 67]:
        # log_acc = (log_p_new - log_q_new) - (log_p_old - log_q_old)
        # 其中 log_p = -S(phi)
        log_acc_ratio = (-prop_s - prop_log_q) - (-curr_s - curr_log_q)

        # 接受判定
        if torch.log(torch.rand(1, device=device)) < log_acc_ratio:
            curr_phi, curr_log_q, curr_s = prop_phi, prop_log_q, prop_s
            accepted_count += 1

        # 将当前状态加入集成（无论是否发生跳转）
        ensemble.append(curr_phi.cpu().numpy())

    print(f"集成生成完毕！最终接受率: {accepted_count / total_n:.2%}")
    return np.array(ensemble)


if __name__ == "__main__":
    # 假设你的模型保存在这里
    PATH = "best_flow_model.pt"
    trained_model = load_trained_model(PATH, CONFIG['L'])

    # 生成 10,000 个构型
    final_configs = produce_ensemble(trained_model, total_n=10000)

    # 保存结果供后续物理分析（如计算 Green's function）
    np.save("phi_ensemble.npy", final_configs)