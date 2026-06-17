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
    # 🌟 核心保护：记录原精度，并动态强制提升至 FP64
    original_dtype = next(model.parameters()).dtype
    model = model.double()
    eval_dtype = torch.float64

    all_phis = []
    all_log_qs = []

    # --- 第一步：并行采样 (GPU 加速) ---
    print(f"正在并行生成 {total_n} 个提案 (已开启双精度 FP64 防截断保护)...")
    with torch.no_grad():
        for _ in range(0, total_n, batch_size):
            current_batch = min(batch_size, total_n - len(all_phis))
            # 🌟 修改：采样噪声 z 时必须指定双精度 dtype=eval_dtype
            z = torch.randn(current_batch, 1, CONFIG['L'], CONFIG['L'], device=device, dtype=eval_dtype)
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

    # 🌟 顺手修个小 Bug：因为第一个直接算作起点，实际提案验证了 total_n - 1 次
    for i in range(1, total_n):
        prop_phi = phi_proposals[i:i + 1]
        prop_log_q = log_q_proposals[i:i + 1]
        prop_s = s_proposals[i:i + 1]

        # log_acc = (log_p_new - log_q_new) - (log_p_old - log_q_old)
        # 其中 log_p = -S(phi)
        log_acc_ratio = (-prop_s - prop_log_q) - (-curr_s - curr_log_q)

        # 🌟 修改：随机数判定也使用双精度 eval_dtype
        if torch.log(torch.rand(1, device=device, dtype=eval_dtype)) < log_acc_ratio:
            curr_phi, curr_log_q, curr_s = prop_phi, prop_log_q, prop_s
            accepted_count += 1

        # 🌟 修改：虽然运算过程是双精度，但存入数组时退回 float32 即可，节约硬盘和内存
        ensemble.append(curr_phi.float().cpu().numpy())

    # 🌟 修改：计算并提取接受率
    acc_rate = accepted_count / (total_n - 1)
    print(f"集成生成完毕！最终接受率: {acc_rate:.2%}")

    # 🌟 评估结束，恢复模型至原训练精度
    if original_dtype == torch.float32:
        model = model.float()

    # 🌟 修改：把 acc_rate 一起 return 出来
    return np.array(ensemble), acc_rate


if __name__ == "__main__":
    # 假设你的模型保存在这里
    # PATH = "best_cnn_model.pt"
    PATH = CONFIG['save_path']

    def convert_path(path):
        # 🌟 修改：打包文件通常用 .npz 后缀
        return path.replace("best_", "ensemble_").replace(".pt", ".npz")

    # 如果你的 CONFIG 里没有 ensemble_path，可以直接用 convert_path
    ensemble_path = CONFIG.get('phi_ensemble_save_path', convert_path(PATH))
    trained_model = load_trained_model(CONFIG)

    # 🌟 修改：接收构型和接受率
    final_configs, final_acc = produce_ensemble(trained_model, total_n=50000)

    # 🌟 修改：使用 np.savez 打包保存！
    # 以后读取时可以用 data = np.load(ensemble_path)
    # 然后 data['configs'] 拿构型，data['acc_rate'] 拿接受率
    np.savez(ensemble_path, configs=final_configs, acc_rate=final_acc)
    print(f"✅ 已成功将构型和接受率打包保存至: {ensemble_path}")