import os
import torch
import importlib
import glob
import shutil

# ==========================================
# 1. 基础配置
# ==========================================
# 需要评估的三个模块名 (对应你的三个 .py 文件名，不要加 .py 后缀)
MODULES = [
    "final_normalizing",
    "step0_final_normalizing",
    "final_normalizing_with_soft_z2"
]

EVAL_SAMPLES = 200000  # MCMC 样本量
START_ITER = 2000
END_ITER = 30000
STEP_SIZE = 2000

# 初始化设备
print(torch.cuda.is_available())
device = torch.device(
    "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
print(f"评估运行设备: {device}")


def evaluate_models():
    for mod_name in MODULES:
        print(f"\n{'=' * 60}")
        print(f"🚀 开始处理模型: {mod_name}")
        print(f"{'=' * 60}")

        try:
            mod = importlib.import_module(mod_name)
            config = mod.CONFIG
            FlowModel = mod.FlowModel
            FreeFieldPrior = mod.FreeFieldPrior
            run_mcmc_evaluation = mod.run_mcmc_evaluation
        except ImportError as e:
            print(f"❌ 导入失败 {mod_name}: {e}")
            continue

        # 实例化模型和先验
        model = FlowModel(config).to(device)
        prior = FreeFieldPrior(L=config['L'], m_sq_prior=0.6005269985).to(device)

        if config.get('double_precision', False):
            model = model.double()
            prior = prior.double()

        base_name = config['base_name']

        # 遍历 2000 到 30000 步的存档
        for step in range(START_ITER, END_ITER + 1, STEP_SIZE):
            pt_path = f"iter_{step}_{base_name}.pt"

            if not os.path.exists(pt_path):
                print(f"⚠️ 找不到存档，跳过: {pt_path}")
                continue

            print(f"\n⏳ 正在加载存档: {pt_path}")
            try:
                checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
                # 清洗字典以防 DDP/Compile 污染
                clean_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                model.load_state_dict(clean_dict)
            except Exception as e:
                print(f"❌ 读取权重失败: {e}")
                continue

            # 判断对称性状态
            enforce_sym = config.get('enforce_z2_sym', False) and (step >= config.get('sym_start_iter', 0))
            z2_status = "ON" if enforce_sym else "OFF"

            print(f"🔬 运行 MCMC 评估 (Samples: {EVAL_SAMPLES}, Z2_Sym: {z2_status})...")

            # 运行你的 MCMC 评估函数
            acc_rate, phi_means, phi_errs = run_mcmc_evaluation(
                model, prior,
                total_n=EVAL_SAMPLES,
                batch_size=config['batch_size'],
                enforce_sym=enforce_sym
            )

            # 格式化新的文件名: 例如 acc_78.52percent_iter_2000_...pt
            acc_str = f"{acc_rate * 100:.2f}".replace('.', 'p')  # 将小数点替换为 p，避免文件名解析问题，也可以直接用 int()
            new_name = f"acc_{acc_str}percent_{pt_path}"

            # 复制并重命名存档
            shutil.copy2(pt_path, new_name)

            print(f"✅ 测试完成! 接受率: {acc_rate:.2%}")
            for p in range(1, 6):
                print(f"   -> phi^{p}: {phi_means[p - 1]:.6f} ± {phi_errs[p - 1]:.6f}")
            print(f"💾 已另存为: {new_name}")


if __name__ == "__main__":
    # 如果训练时使用了高精度，这里统一打开
    # torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.benchmark = True
    evaluate_models()