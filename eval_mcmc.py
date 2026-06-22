import torch
import numpy as np
import os
import json

# 直接从你的主训练脚本（假设文件名为 var_shared_trunk_prior.py）中导入所需模块
from var_shared_trunk_prior import (
    CONFIG, FlowModel, FreeFieldPrior, run_mcmc_evaluation, device
)

# 👇 🌟 新增：从外部强行把拉普拉斯核转换为双精度！
import var_shared_trunk_prior
# var_shared_trunk_prior.laplacian_kernel = var_shared_trunk_prior.laplacian_kernel.double()


def evaluate_milestone_checkpoints():
    # 1. 物理测试参数
    total_n = 200000
    batch_size = 20000  # 可根据你的显存大小(VRAM)适度上调，L=14 时 1M 样本约占 800MB 显存
    target_rates = [40, 45, 50, 55, 60]

    # 2. 提取基础模型名称并修复 CONFIG
    # 👇 🌟 强制 MCMC 过程使用双精度
    CONFIG['double_precision'] = False
    # torch.set_default_dtype(torch.float64)  # 设定全局张量默认类型为 FP64

    # 修复结构参数
    CONFIG['t_head_channels'] = [48] * 6
    CONFIG['s_head_channels'] = [96] * 6
    CONFIG['trunk_channels'] = [64] * 6

    base_name = "shared_trunk_prior_cnn_dp_False_L14_c6_d2_TrCh64x6_Ly3x6_trk_3_3_dil_1_2_Sh96x6L1x6_Th48x6L1x6_iter_60000"

    # 3. 初始化自由场先验 (显式调用 .double())
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=abs(CONFIG['m_sq'])).to(device)
    if CONFIG.get('double_precision', False):
        prior = prior.double()

    all_results = {}

    print(f"🌟 准备开始 MCMC 评估任务，总构型数: {total_n}")

    # 4. 循环测试不同接受率的 Checkpoint
    for rate in target_rates:
        ckpt_name = f"acc_{rate}percent_{base_name}.pt"

        if not os.path.exists(ckpt_name):
            print(f"\n⚠️ 找不到权重文件: {ckpt_name}，已跳过。")
            continue

        print(f"\n{'=' * 60}")
        print(f"🚀 开始评估里程碑模型: {rate}% 目标接受率")
        print(f"📁 加载权重: {ckpt_name}")

        # 每次都重新初始化干净的模型，并显式转为双精度
        model = FlowModel(CONFIG).to(device)

        # 加载 FP32 权重，PyTorch 会自动将其映射到模型的 FP64 参数上
        checkpoint = torch.load(ckpt_name, map_location=device, weights_only=False)
        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(clean_dict)
        model.eval()

        # 判断对称性开关 (如果是 iter_60000 且开启了 enforce_z2_sym，则这里应为 True)
        checkpoint_iter = checkpoint.get('iteration', 60000)
        enforce_sym = CONFIG.get('enforce_z2_sym', False) and (checkpoint_iter >= CONFIG.get('sym_start_iter', 0))

        print(f"✅ 模型加载成功 (Enforce Z2 Sym: {enforce_sym})。开始生成并执行 MCMC...")

        # 🚀 核心：直接调用你写好的在线评估引擎
        acc_rate, phi_means, phi_errs = run_mcmc_evaluation(
            model,
            prior,
            total_n=total_n,
            batch_size=batch_size,
            enforce_sym=enforce_sym
        )

        print(f"📊 实际 MCMC 接受率: {acc_rate:.2%}")
        print("------ Phi Powers Expectation ------")
        for p in range(1, 6):
            # 使用科学计数法打印，对齐数据
            print(f"phi^{p}: {phi_means[p - 1]:+10.6e} ± {phi_errs[p - 1]:.6e}")

        # 5. 存储当次结果
        all_results[f"acc_{rate}"] = {
            "target_acc": rate,
            "actual_acc": acc_rate,
            "phi_means": phi_means,
            "phi_errs": phi_errs
        }

    # 6. 保存所有结果至硬盘
    save_prefix = f"evaluation_1M_observables_{base_name}"

    # 保存为 Numpy 格式，方便后续画图读取
    np.save(save_prefix + ".npy", all_results)

    # 保存为 JSON 格式，方便人类直接查看
    with open(save_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n🎉 评估全部完成！数据已保存至: \n 1. {save_prefix}.npy \n 2. {save_prefix}.json")


if __name__ == "__main__":
    evaluate_milestone_checkpoints()