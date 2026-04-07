import torch
import numpy as np
from normalizing_flow import RealNVP, compute_action, CONFIG, device


def load_trained_model(config):
    """加载保存的模型权重"""
    model = RealNVP(config=config).to(device)
    # 根据你之前训练脚本保存的文件名动态加载
    model_path = config['model_save_path']

    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"警告: 找不到模型文件 {model_path}，请确认是否已训练完成！")
        return None

    model.eval()
    return model


def test_model(batch_size, iterations):
    for config in CONFIG:
        print(f"\n========== 开始测试 L={config['L']} ==========")
        model = load_trained_model(config)
        if model is None:
            continue

        # 存放最终集成的列表（放在 CPU 内存中）
        ensemble = []
        accepted_count = 0
        total_proposals = batch_size * iterations

        print(f"正在构建 {batch_size} 条并行的马尔可夫链 (共需生成 {total_proposals} 个样本)...")

        # 核心修复 1：必须关闭梯度计算
        with torch.no_grad():
            # 初始化 1024 条链的起点
            z_init = torch.randn(batch_size, config['L'], config['L'], device=device)
            curr_phi, curr_log_q = model(z_init)
            curr_s = compute_action(curr_phi, config)  # 核心修复 2：补上 config

            for i in range(iterations):
                # 1. 并行生成一批提议构型
                z_prop = torch.randn(batch_size, config['L'], config['L'], device=device)
                prop_phi, prop_log_q = model(z_prop)
                prop_s = compute_action(prop_phi, config)

                # 2. 并行计算 1024 个接受率对数
                # log_A = log_q(phi) + S(phi) - log_q(phi') - S(phi')
                log_acc_ratio = curr_log_q + curr_s - prop_log_q - prop_s

                # 3. 接受判定
                log_u = torch.log(torch.rand(batch_size, device=device))
                accept_mask = log_u < log_acc_ratio  # shape: [1024]

                # 4. 状态更新 (利用 torch.where 并行更新 1024 条链)
                mask_phi = accept_mask.view(-1, 1, 1)  # 变形以广播到 [batch, L, L]

                curr_phi = torch.where(mask_phi, prop_phi, curr_phi)
                curr_log_q = torch.where(accept_mask, prop_log_q, curr_log_q)
                curr_s = torch.where(accept_mask, prop_s, curr_s)

                # 5. 统计与存储
                current_acc_sum = accept_mask.sum().item()
                accepted_count += current_acc_sum

                # 将这 1024 个当前状态拷贝回 CPU 并保存
                ensemble.append(curr_phi.cpu().numpy())

                if (i + 1) % 1000 == 0:
                    print(f"已推进 {i + 1}/{iterations} 步 | 最近一批接受率: {current_acc_sum / batch_size:.2%}")

        # 将所有的 list 拼成一个巨大的 numpy 数组
        ensemble_np = np.concatenate(ensemble, axis=0)
        final_acc_rate = accepted_count / total_proposals

        print(f"L={config['L']} 集成生成完毕！")
        print(f"最终集成形状: {ensemble_np.shape} | 总体平均接受率: {final_acc_rate:.2%}")

        # 保存到本地，供下一步物理量计算使用
        np.save(f"flow_ensemble_L{config['L']}.npy", ensemble_np)


if __name__ == "__main__":
    # 为了测试，建议先用 1024 batch_size 和 1000 iterations 跑个 100 万样本看看
    test_model(1024, 1000)