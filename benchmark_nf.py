import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
from torch.profiler import profile, ProfilerActivity
from HMC_final import CONFIG as HMC_CONFIG_LIST
from final_normalizing import FlowModel, FreeFieldPrior, CONFIG as NF_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==============================================================================
# 🎛️ 核心参数与权重锁定
# ==============================================================================
TARGET_L = 14
TARGET_TIME_SECONDS = 180.0
NF_MAX_BATCH = 163_840

# 🌟 真实物理产出指标
TRUE_NF_ACCEPTANCE = 0.70

# 💎 锁定你辛辛苦苦训练出的巅峰模型文件
SPECIFIC_CKPT_PATH = "acc_70percent_final_normalizing_32_dp_False_L14_c12_d2_TrCh32x12_Ly3x12_trk_3_dil_1_Sh32x12L1x12_Th32x12L1x12_iter_100000.pt"

# --- 环境准备 ---
hmc_config = next(c for c in HMC_CONFIG_LIST if c.get('L') == TARGET_L)
dtype = torch.float32

NF_CONFIG['L'] = TARGET_L
NF_CONFIG['m_sq'] = hmc_config['m2']
NF_CONFIG['lam'] = hmc_config['lam']
NF_CONFIG['double_precision'] = False

# --- 阶段一：模型加载与真实 FLOPs 追踪 ---
print(f"{'=' * 85}")
print(f"🔥 Normalizing Flow 独立拉力赛：巅峰模型算力压测 (限时 {TARGET_TIME_SECONDS}秒)")
print(f"💻 设备: {torch.cuda.get_device_name(0)} | 精度: Float32")
print(f"{'=' * 85}")

nf_model = FlowModel(NF_CONFIG).to(device)
nf_prior = FreeFieldPrior(L=TARGET_L, m_sq_prior=0.6005269985).to(device)
dummy_prog = torch.tensor(1.0, device=device, dtype=dtype)
enforce_sym = NF_CONFIG.get('enforce_z2_sym', False)

# 💉 注入物理灵魂：精准加载巅峰模型
print(f"\n[加载中] 正在加载指定模型权重: {SPECIFIC_CKPT_PATH}")
try:
    if os.path.exists(SPECIFIC_CKPT_PATH):
        checkpoint = torch.load(SPECIFIC_CKPT_PATH, map_location=device)
        nf_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"    ✅ 成功加载巅峰权重！灵魂注入完毕。")
    else:
        print(f"    ❌ 找不到文件 {SPECIFIC_CKPT_PATH}，请确保它和本脚本在同一目录下！")
        exit(1)
except Exception as e:
    print(f"    ❌ 权重加载失败: {e}")
    exit(1)

nf_model.eval()

print("\n[追踪中] 正在探测 Y-Net 前向传播的真实浮点运算量...")
TRACE_BATCH = 16
for _ in range(3):
    z, _ = nf_prior.sample(TRACE_BATCH)
    nf_model(z, dummy_prog, enforce_sym=enforce_sym)

z, _ = nf_prior.sample(TRACE_BATCH)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
    nf_model(z, dummy_prog, enforce_sym=enforce_sym)

total_flops = sum([evt.flops for evt in prof.events() if evt.flops is not None and evt.flops > 0])
real_nf_flops = total_flops / TRACE_BATCH
print(f"    🎯 NF 单次动作 (1个样本) 真实算力: {real_nf_flops:,.0f} FLOPs")

# --- 阶段二：满载 3 分钟拉力赛 ---
print(f"\n[比赛进行中] 🏎️ NF 正在满显存全力狂奔 (目标耗时 {TARGET_TIME_SECONDS} 秒)...")

with torch.no_grad():
    for _ in range(5):
        z, _ = nf_prior.sample(NF_MAX_BATCH)
        nf_model(z, dummy_prog, enforce_sym=enforce_sym)
    torch.cuda.synchronize()
    nf_start_time = time.perf_counter()
    nf_loops, nf_total_raw_samples = 0, 0

    while True:
        z, _ = nf_prior.sample(NF_MAX_BATCH)
        nf_model(z, dummy_prog, enforce_sym=enforce_sym)
        nf_loops += 1
        nf_total_raw_samples += NF_MAX_BATCH
        torch.cuda.synchronize()
        nf_elapsed = time.perf_counter() - nf_start_time
        if nf_elapsed >= TARGET_TIME_SECONDS: break

# --- 阶段三：NF 成绩单 ---
nf_actions_per_sec = nf_total_raw_samples / nf_elapsed
actual_nf_tflops = (nf_actions_per_sec * real_nf_flops) / 1e12
nf_effective_samples_total = nf_total_raw_samples * TRUE_NF_ACCEPTANCE
nf_effective_samples_per_sec = nf_effective_samples_total / nf_elapsed

print("\n" + "=" * 85)
print("🏁 Normalizing Flow 独立拉力赛最终成绩")
print("=" * 85)
print(f"📊 维度一：底层算力压榨 (Compute-bound vs Memory-bound)")
print(f"  - 真实硬件动作吞吐: {nf_actions_per_sec:,.0f} 次前向传播/秒")
print(f"  - 实际榨取物理算力: {actual_nf_tflops:.3f} TFLOPs/s")

print(f"\n📊 维度二：{TARGET_TIME_SECONDS}秒真实时间内的物理总账")
print(f"  - 总计运行时间: {nf_elapsed:.2f} 秒")
print(f"  - 并发规模: {NF_MAX_BATCH:,} 链")
print(f"  - 生成原始样本: {nf_total_raw_samples:,} 个")
print(f"  - 真实 M-H 接受率: {TRUE_NF_ACCEPTANCE * 100:.1f}%")
print(f"  - 👉 折算独立样本: {nf_effective_samples_total:,.2f} 个")
print(f"  - 👉 NF 物理级生成效率: {nf_effective_samples_per_sec:,.2f} 独立样本/秒")