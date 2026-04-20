import torch
import numpy as np
from cnn_res_net import FlowModel, compute_action, CONFIG, device  # 复用你之前的定义


def load_trained_model(checkpoint_path, L, coupling_layers=16, hidden_channels=16,
                      num_hidden_layers=12):
    """加载保存的模型权重"""
    model = FlowModel(L=L,coupling_layers=coupling_layers,
                      hidden_channels=hidden_channels,
                      num_hidden_layers=num_hidden_layers
                      ).to(device)
    if CONFIG.get('double precision', False): # 开启双精度
        # 显式转换为双精度
        model = model.double()
    # map_location 确保在没有 GPU 的机器上也能加载
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 如果保存的是整个 dict (包含 optimizer)，则取 model_state_dict
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()  # 必须切换到评估模式
    return model


def produce_ensemble(model, total_n=100000, batch_size=1024):
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
    # 【修改点 1】预分配 Tensor 存储物理构型，避免列表 append 导致显存/内存碎片化
    ensemble = torch.empty((total_n, 1, CONFIG['L'], CONFIG['L']), dtype=phi_proposals.dtype)

    # 【修改点 2】专门预分配一个布尔型 Tensor，记录每一步是否发生了跳转
    accept_history = torch.empty(total_n, dtype=torch.bool)
    # 第 0 步是链的起点，默认算作在当前位置 (True)
    ensemble[0] = curr_phi.cpu()
    accept_history[0] = True
    accepted_count = 1
    for i in range(1, total_n):
        prop_phi = phi_proposals[i:i + 1]
        prop_log_q = log_q_proposals[i:i + 1]
        prop_s = s_proposals[i:i + 1]

        # log_acc = (log_p_new - log_q_new) - (log_p_old - log_q_old)
        # 其中 log_p = -S(phi)
        log_acc_ratio = (-prop_s - prop_log_q) - (-curr_s - curr_log_q)

        # 【修改点 3】显式提取本次的接受判定结果 (True / False)
        is_accepted = torch.log(torch.rand(1, device=device)) < log_acc_ratio

        # 接受判定
        if is_accepted:
            curr_phi, curr_log_q, curr_s = prop_phi, prop_log_q, prop_s
            accepted_count += 1

        # 【修改点 4】同步将当前构型和接受状态写入预分配的内存中
        ensemble[i] = curr_phi.cpu()
        accept_history[i] = is_accepted.cpu().squeeze()

    print(f"集成生成完毕！最终接受率: {accepted_count / total_n:.2%}")
    return ensemble.numpy(), accept_history.numpy()


if __name__ == "__main__":
    # 假设你的模型保存在这里
    # PATH = "best_cnn_model.pt"
    PATH = CONFIG['save_path']
    print(PATH)
    if CONFIG.get('double precision', False): # 开启双精度
        torch.set_default_dtype(torch.float64)
        print('double precision: True')
    trained_model = load_trained_model(PATH, CONFIG['L'],
                                       coupling_layers=CONFIG['coupling_layers'],
                                       hidden_channels=CONFIG['hidden_channels'],
                                        num_hidden_layers=CONFIG['hidden_layers'])

    # 生成 10,0000 个构型
    final_configs, accept_traj = produce_ensemble(trained_model, total_n=1000000)
    # 【优雅的保存方式】将物理构型和MH判定历史打包保存在同一个文件里
    save_file = CONFIG['phi_ensemble_save_path']
    np.savez_compressed(
        save_file,
        configs=final_configs,
        accept_history=accept_traj
    )
    print(f"数据已打包保存至 {save_file}")

    # 后续你在做数据分析脚本时，只需要这样读取：
    # data = np.load("phi_ensemble_with_history.npz")
    # loaded_configs = data['configs']
    # loaded_history = data['accept_history']
