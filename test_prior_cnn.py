import torch
import numpy as np
from prior_cnn_multi_kernel import FlowModel, FreeFieldPrior, compute_action, CONFIG, device
# from prior_cnn_net import FlowModel, FreeFieldPrior, compute_action, CONFIG, device


def load_trained_model(checkpoint_path):
    """加载保存的模型权重"""
    model = FlowModel(
        CONFIG
    ).to(device)

    if CONFIG.get('double precision', False):
        model = model.double()

    # map_location 确保在不同硬件环境下顺利加载
    state_dict = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in state_dict:
        raw_dict = state_dict['model_state_dict']
    else:
        raw_dict = state_dict

        # ==========================================
        # 【新增代码】清洗权重字典，剥离 torch.compile 的包装前缀
        # ==========================================
    clean_dict = {}
    for key, value in raw_dict.items():
        # 如果前缀包含 _orig_mod.，直接把它替换为空字符串
        clean_key = key.replace('_orig_mod.', '')
        clean_dict[clean_key] = value
    # ==========================================

    # 使用清洗后的干净字典加载权重
    model.load_state_dict(clean_dict)

    model.eval()  # 切换到评估模式
    return model


def produce_ensemble(model, prior, total_n=100000, batch_size=1024):
    """
    批量生成物理构型集成
    1. 并行生成 Proposal (从自由场先验采样 -> 流模型变形)
    2. 串行构建 Markov Chain (MH 验证)
    """
    all_phis = []
    all_log_qs = []

    # --- 第一步：并行采样 (GPU 加速) ---
    print(f"正在并行生成 {total_n} 个提案...")
    with torch.no_grad():
        for _ in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - len(all_phis))

            # 1. 从自由场先验中采样
            z, log_p_z = prior.sample(current_batch)

            # 2. 通过流模型进行微调变形，获取雅可比行列式对数
            phi, log_det_J = model(z)

            # 3. 计算最终的生成概率密度 log q(phi)
            log_q = log_p_z - log_det_J

            all_phis.append(phi)
            all_log_qs.append(log_q)

    phi_proposals = torch.cat(all_phis, dim=0)
    log_q_proposals = torch.cat(all_log_qs, dim=0)

    # 提前计算所有提案的 Action S(phi)
    s_proposals = compute_action(phi_proposals)

    # --- 第二步：串行 MH 验证 (构建马尔可夫链) ---
    print("正在通过 MH 算法构建马尔可夫链...")

    # 初始化链的第一个状态
    curr_phi = phi_proposals[0:1]
    curr_log_q = log_q_proposals[0:1]
    curr_s = s_proposals[0:1]

    # 预分配 Tensor 存储物理构型，避免列表 append 导致内存碎片化
    ensemble = torch.empty((total_n, 1, CONFIG['L'], CONFIG['L']), dtype=phi_proposals.dtype)

    # 预分配布尔型 Tensor，记录每一步是否发生了跳转
    accept_history = torch.empty(total_n, dtype=torch.bool)

    # 第 0 步是链的起点
    ensemble[0] = curr_phi.cpu()
    accept_history[0] = True
    accepted_count = 1

    for i in range(1, total_n):
        prop_phi = phi_proposals[i:i + 1]
        prop_log_q = log_q_proposals[i:i + 1]
        prop_s = s_proposals[i:i + 1]

        # 计算 Metropolis-Hastings 接受率对数
        # log_acc = (-S_new - log_q_new) - (-S_old - log_q_old)
        log_acc_ratio = (-prop_s - prop_log_q) - (-curr_s - curr_log_q)

        # 显式提取本次的接受判定结果
        is_accepted = torch.log(torch.rand(1, device=device)) < log_acc_ratio

        if is_accepted:
            curr_phi, curr_log_q, curr_s = prop_phi, prop_log_q, prop_s
            accepted_count += 1

        if i % 100 == 0:
            print(f'step:{i},accept ratio:{accepted_count/i:.2%}')

        # 同步写入预分配的内存中
        ensemble[i] = curr_phi.cpu()
        accept_history[i] = is_accepted.cpu().squeeze()

    print(f"集成生成完毕！最终接受率: {accepted_count / total_n:.2%}")
    return ensemble.numpy(), accept_history.numpy()


if __name__ == "__main__":
    PATH = CONFIG['save_path']
    # PATH = 'best_prior_cnn_res_model_double_precision_True_14_coupling_layers_32_hidden_layers_6_hidden_channels_16_iterations_45000.pt'
    print(f"模型加载路径: {PATH}")

    if CONFIG.get('double precision', False):
        torch.set_default_dtype(torch.float64)
        print('double precision: True')

    trained_model = load_trained_model(
        PATH
    )

    # --- 实例化并配置 FreeFieldPrior ---
    # 破缺相下使用质量的绝对值作为先验的正质量参数
    prior_m_sq = abs(CONFIG['m_sq'])
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=prior_m_sq).to(device)

    if CONFIG.get('double precision', False):
        prior = prior.double()

    # 生成物理集成 (例如 100,000 个构型)
    final_configs, accept_traj = produce_ensemble(trained_model, prior, total_n=100000)

    # 打包保存
    save_file = CONFIG['phi_ensemble_save_path']
    np.savez_compressed(
        save_file,
        configs=final_configs,
        accept_history=accept_traj
    )
    print(f"数据已打包保存至 {save_file}")