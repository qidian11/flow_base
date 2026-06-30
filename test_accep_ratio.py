import torch
import numpy as np
import os

# 直接从你的训练脚本中导入所有需要的组件
from final_normalizing import (
    FlowModel,
    FreeFieldPrior,
    run_mcmc_evaluation,
    CONFIG,
    device
)

if __name__ == "__main__":
    # ===============================
    # ⬇️ 只需要在这里修改测试配置 ⬇️
    # ===============================
    CHECKPOINT_FILE = "iter_6000_final_normalizing_32_dp_False_L14_c12_d2_TrCh32x12_Ly3x12_trk_3_dil_1_Sh32x12L1x12_Th32x12L1x12_iter_100000.pt"
    TEST_TOTAL_N = 100000
    TEST_BATCH_SIZE = 1024
    TEST_ENFORCE_SYM = False
    # ===============================
    print("=" * 50)
    print(f"正在测试：{CHECKPOINT_FILE}")
    print("=" * 50)

    if not os.path.exists(CHECKPOINT_FILE):
        print(f"❌ 找不到权重文件: {CHECKPOINT_FILE}")
        exit(1)

    if CONFIG.get('double_precision', False):
        torch.set_default_dtype(torch.float64)

    print(f"🔧 正在初始化模型架构并导入权重...")

    # 实例化模型和先验（它们会自动使用 final_normalizing.py 中的全局 CONFIG）
    model = FlowModel(CONFIG).to(device)
    prior = FreeFieldPrior(L=CONFIG['L'], m_sq_prior=0.6005269985).to(device)

    if CONFIG['double_precision']:
        model = model.double()
        prior = prior.double()

    # 加载权重
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(clean_dict)

    # 模拟训练末期：彻底打开防爆盾限制
    model.step_warmup(1.0)
    model.eval()

    print(f"\n🚀 开始执行 MCMC 评估 (总样本数: {TEST_TOTAL_N}, 批大小: {TEST_BATCH_SIZE})...")

    # 调用原脚本的评估函数
    acc_rate, phi_means, phi_errs = run_mcmc_evaluation(
        model, prior,
        total_n=TEST_TOTAL_N,
        batch_size=TEST_BATCH_SIZE,
        enforce_sym=TEST_ENFORCE_SYM
    )

    print("\n" + "=" * 50)
    print(f"✅ 测试完成!")
    print(f"📊 物理接受率: {acc_rate:.2%}")
    print("------ Phi Powers Expectation ------")
    for p in range(1, 6):
        print(f"phi^{p}: {phi_means[p - 1]:.6f} ± {phi_errs[p - 1]:.6f}")
    print("=" * 50)