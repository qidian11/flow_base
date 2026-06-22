import torch
import numpy as np
import os
import time

# 🌟 修改 1：确保从你最新的主网络文件中导入！
from var_shared_trunk_prior import FlowModel, FreeFieldPrior, compute_action, CONFIG, device


def load_trained_model(checkpoint_path):
    """加载保存的模型权重"""
    model = FlowModel(CONFIG).to(device)

    if CONFIG.get('double_precision', False):
        model = model.double()

    # map_location 确保在不同硬件环境下顺利加载
    state_dict = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in state_dict:
        raw_dict = state_dict['model_state_dict']
    else:
        raw_dict = state_dict

    # 清洗权重字典，剥离 torch.compile 的包装前缀
    clean_dict = {}
    for key, value in raw_dict.items():
        clean_key = key.replace('_orig_mod.', '')
        clean_dict[clean_key] = value

    model.load_state_dict(clean_dict)
    model.eval()  # 切换到评估模式
    return model


def produce_ensemble(model, prior, total_n=100000, batch_size=1024):
    """
    批量生成物理构型集成
    修复了 OOM 显存灾难，并批处理了 Action 计算
    """
    original_dtype = next(model.parameters()).dtype

    # 保持双精度以满足你的物理验证需求
    model = model.double()
    prior = prior.double()
    eval_dtype = torch.float64

    all_phis = []
    all_log_qs = []

    dummy_progress = torch.tensor(1.0, device=device, dtype=eval_dtype)
    enforce_sym = CONFIG.get('enforce_z2_sym', False)

    # --- 第一步：并行采样 (GPU 加速) ---
    print(f"[{time.strftime('%H:%M:%S')}] 开始生成 {total_n} 个提案 (FP64 模式)...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - len(all_phis))

            z, log_p_z = prior.sample(current_batch)
            phi, log_det_J = model(z, progress=dummy_progress, enforce_sym=enforce_sym)
            log_q = log_p_z - log_det_J

            all_phis.append(phi)
            all_log_qs.append(log_q)

            if (i // batch_size) % 10 == 0:
                print(
                    f"   [采样中] 已生成: {min(i + batch_size, total_n)} / {total_n} (耗时: {time.time() - start_time:.2f}s)")

    phi_proposals = torch.cat(all_phis, dim=0)
    log_q_proposals = torch.cat(all_log_qs, dim=0)
    print(f"[{time.strftime('%H:%M:%S')}] 提案生成完毕！准备计算 Action...")

    # --- 第二步：批处理计算 Action (彻底解决 VRAM 溢出卡死) ---
    print(f"[{time.strftime('%H:%M:%S')}] 正在分批计算 Action (防止显存爆炸)...")
    s_proposals_list = []
    action_start = time.time()

    with torch.no_grad():
        # 这里复用 batch_size，或者为了显存安全可以适当调小
        action_batch_size = max(256, batch_size // 2)
        for i in range(0, total_n, action_batch_size):
            batch_phi = phi_proposals[i:i + action_batch_size]
            s_batch = compute_action(batch_phi)
            s_proposals_list.append(s_batch)

            if (i // action_batch_size) % 50 == 0 and i > 0:
                print(f"   [Action 计算] 已处理: {i} / {total_n}")

    s_proposals = torch.cat(s_proposals_list, dim=0)
    print(f"[{time.strftime('%H:%M:%S')}] Action 计算完毕！(耗时: {time.time() - action_start:.2f}s)")

    # --- 第三步：串行 MH 验证 (CPU NumPy 极速处理) ---
    print(f"[{time.strftime('%H:%M:%S')}] 开始 CPU NumPy 马尔可夫链筛选...")
    mh_start = time.time()

    s_np = s_proposals.cpu().numpy()
    log_q_np = log_q_proposals.cpu().numpy()
    log_rands_np = np.log(np.random.rand(total_n))

    accepted_indices = np.zeros(total_n, dtype=int)
    accept_history = np.zeros(total_n, dtype=bool)

    accepted_count = 1
    accepted_indices[0] = 0
    accept_history[0] = True

    curr_s_val = s_np[0]
    curr_log_q_val = log_q_np[0]

    for i in range(1, total_n):
        prop_s_val = s_np[i]
        prop_log_q_val = log_q_np[i]

        log_acc_ratio = (-prop_s_val - prop_log_q_val) - (-curr_s_val - curr_log_q_val)

        if log_rands_np[i] < log_acc_ratio:
            curr_s_val = prop_s_val
            curr_log_q_val = prop_log_q_val
            accepted_indices[i] = i
            accept_history[i] = True
            accepted_count += 1
        else:
            accepted_indices[i] = accepted_indices[i - 1]
            accept_history[i] = False

    print(f"[{time.strftime('%H:%M:%S')}] MH 筛选完毕！纯 CPU 计算耗时: {time.time() - mh_start:.4f}s")

    # 一次性切片拉取最终链，并转换回 float32 节省硬盘空间
    idx_tensor = torch.tensor(accepted_indices, device=device, dtype=torch.long)
    ensemble = phi_proposals[idx_tensor].float().cpu().numpy()

    print(f"🎉 集成生成彻底完成！最终接受率: {accepted_count / total_n:.2%}，总耗时: {time.time() - start_time:.2f}s")

    if original_dtype == torch.float32:
        model = model.float()

    return ensemble, accept_history


if __name__ == "__main__":
    PATH = CONFIG['save_path']
    # 如果想手动指定模型，可以在这里解除注释：
    # PATH = 'latest_shared_trunk_prior_cnn_dp_False_L14_c6_d2_TrCh64x6_Ly3x6_trk_3_3_dil_1_2_Sh96x6L1x6_Th48x6L1x6_iter_60000.pt'

    print(f"模型加载路径: {PATH}")

    if CONFIG.get('double_precision', False):
        torch.set_default_dtype(torch.float64)
        print('double precision: True')

    trained_model = load_trained_model(PATH)

    # --- 实例化并配置 FreeFieldPrior ---
    # 破缺相下使用质量的绝对值作为先验的正质量参数
    prior_m_sq = abs(CONFIG['m_sq'])
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=prior_m_sq).to(device)

    if CONFIG.get('double_precision', False):
        prior = prior.double()

    # 生成物理集成 (100,000 个构型)
    final_configs, accept_traj = produce_ensemble(trained_model, prior, total_n=200000)

    # 打包保存
    save_file = CONFIG['phi_ensemble_save_path']
    np.savez_compressed(
        save_file,
        configs=final_configs,
        accept_history=accept_traj
    )
    print(f"数据已极速打包并压缩保存至 {save_file}")