import torch
import numpy as np
from prior_cnn_macro_z2 import FlowModel, compute_action, CONFIG, device  # 复用你之前的定义


def load_trained_model(config):
    """加载保存的模型权重"""
    model = FlowModel(config).to(device)

    # 直接通过 config 获取路径并读取
    file_path = config['save_path']
    state_dict = torch.load(file_path, map_location=device)

    # 如果保存的是整个 dict (包含 optimizer)，则取 model_state_dict
    if 'model_state_dict' in state_dict:
        # 你之前的代码中将 state_dict 中包含 '_orig_mod.' 的 key 做了清理，
        # 如果你这里保存的是 torch.compile 后的权重，可能也需要像 prior_cnn_macro_z2.py
        # 第 344 行那样做一下 key 的清理，否则直接 load 即可：
        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict['model_state_dict'].items()}
        model.load_state_dict(clean_dict)
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
    # PATH = "best_cnn_model.pt"
    PATH = CONFIG['save_path']


    def convert_path(path):
        return path.replace("best_", "ensemble_").replace(".pt", ".npy")
    ensemble_path = convert_path(PATH)
    trained_model = load_trained_model(CONFIG)

    # 生成 10,000 个构型
    final_configs = produce_ensemble(trained_model, total_n=50000)

    # 保存结果供后续物理分析（如计算 Green's function）
    np.save(ensemble_path, final_configs)