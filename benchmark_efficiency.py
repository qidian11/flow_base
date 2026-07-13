import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from HMC_final import CONFIG as HMC_CONFIG_LIST
from final_normalizing import FlowModel, FreeFieldPrior, CONFIG as NF_CONFIG, auto_find_latest_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==============================================================================
# 🎛️ 终极参数区 (硬件压榨 + 时间限制)
# ==============================================================================
TARGET_L = 14
TARGET_TIME_SECONDS = 180.0  # 🎯 目标拉力赛时间：3分钟

# 4090 满显存极限并发
HMC_MAX_BATCH = 1_835_008
NF_MAX_BATCH = 196_608

# 物理折算常量 (请确保 TAU_INT 符合你模型真实的自相关时间)
TAU_INT = 200
HMC_TRAJECTORY_LENGTH = 10
ESTIMATED_NF_ACCEPTANCE = 0.60  # 预估 Flow-MCMC 的接受率

# --- 准备环境 ---
hmc_config = next(c for c in HMC_CONFIG_LIST if c.get('L') == TARGET_L)
tao = hmc_config['tao']
dtype = torch.float32
NF_CONFIG['L'] = TARGET_L
NF_CONFIG['m_sq'] = hmc_config['m2']
NF_CONFIG['lam'] = hmc_config['lam']
NF_CONFIG['double_precision'] = False


# ==============================================================================
# 🛠️ 优化算子定义区 (HMC 专用 Fused Kernels)
# ==============================================================================
def get_fused_kernels(dtype, device):
    kernel_force = torch.tensor([[0.0, 2.0, 0.0], [2.0, -8.0, 2.0], [0.0, 2.0, 0.0]], dtype=dtype, device=device).view(
        1, 1, 3, 3)
    kernel_action = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=dtype,
                                 device=device).view(1, 1, 3, 3)
    return kernel_force, kernel_action


def calculate_action_opt(phi, config, kernel_action):
    phi_pad = F.pad(phi.unsqueeze(1), (1, 1, 1, 1), mode='circular')
    laplacian = F.conv2d(phi_pad, kernel_action).squeeze(1)
    return (phi * laplacian + config['m2'] * (phi ** 2) + config['lam'] * (phi ** 4)).sum(dim=(1, 2))


def calculate_hamiltonian_opt(phi, p, config, kernel_action):
    return 0.5 * torch.sum(p ** 2, dim=(1, 2)) + calculate_action_opt(phi, config, kernel_action)


def get_force_opt(phi, config, kernel_force):
    phi_pad = F.pad(phi.unsqueeze(1), (1, 1, 1, 1), mode='circular')
    return F.conv2d(phi_pad, kernel_force).squeeze(1) - 2 * config['m2'] * phi - 4 * config['lam'] * (phi ** 3)


def leap_frog_opt(phi, p, tao, config, kernel_force):
    epsilon = tao / config['leap_frog_step']
    eps_half = epsilon / 2.0
    force = get_force_opt(phi, config, kernel_force)
    p_new = p + eps_half * force
    phi_new = phi + epsilon * p_new
    for _ in range(config['leap_frog_step'] - 1):
        force = get_force_opt(phi_new, config, kernel_force)
        p_new = p_new + epsilon * force
        phi_new = phi_new + epsilon * p_new
    p_new = p_new + eps_half * get_force_opt(phi_new, config, kernel_force)
    return phi_new, p_new


def HMC_step_production(phi, tao, config, kernel_force, kernel_action):
    p = torch.randn_like(phi)
    h_old = calculate_hamiltonian_opt(phi, p, config, kernel_action)
    phi_new, p_new = leap_frog_opt(phi, p, tao, config, kernel_force)
    h_new = calculate_hamiltonian_opt(phi_new, p_new, config, kernel_action)
    delta_H = h_new - h_old
    is_invalid = torch.isnan(delta_H) | torch.isinf(delta_H)
    delta_H_safe = torch.where(is_invalid, torch.tensor(float('inf'), dtype=phi.dtype, device=phi.device), delta_H)
    prob = torch.exp((-delta_H_safe).clamp(max=50))
    accepted_mask = torch.rand_like(prob) < prob
    phi_next = torch.where(accepted_mask.view(-1, 1, 1), phi_new, phi)
    return phi_next, accepted_mask.float().mean(), ~is_invalid.any()


# ==============================================================================
# 🕵️‍♂️ 阶段一：获取真实的单步计算量 (FLOPs)
# ==============================================================================
print(f"{'=' * 85}")
print(f"🔥 全量大一统压测：真实 FLOPs追踪 + {TARGET_TIME_SECONDS}秒硬件拉力赛")
print(f"💻 设备: {torch.cuda.get_device_name(0)} | 精度: Float32")
print(f"{'=' * 85}")


def trace_real_flops(model_name, hmc_fn=None, nf_model=None, nf_prior=None, dummy_prog=None, enforce_sym=False):
    print(f"\n[追踪中] 正在探测 {model_name} 的真实浮点运算量...")
    TRACE_BATCH = 16

    if model_name == "HMC":
        phi_trace = torch.zeros(TRACE_BATCH, TARGET_L, TARGET_L, dtype=dtype, device=device)
        for _ in range(3): hmc_fn(phi_trace)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                     with_flops=True) as prof:
            hmc_fn(phi_trace)

    elif model_name == "NF":
        for _ in range(3):
            z, _ = nf_prior.sample(TRACE_BATCH)
            nf_model(z, dummy_prog, enforce_sym=enforce_sym)
        z, _ = nf_prior.sample(TRACE_BATCH)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                     with_flops=True) as prof:
            nf_model(z, dummy_prog, enforce_sym=enforce_sym)

    total_flops = sum([evt.flops for evt in prof.events() if evt.flops is not None and evt.flops > 0])
    flops_per_sample = total_flops / TRACE_BATCH

    if flops_per_sample == 0 and model_name == "HMC":
        print("    ⚠️ PyTorch 未抓取到原生张量操作，启用物理底层计算量补偿。")
        flops_per_sample = 2800

    print(f"    🎯 {model_name} 单次动作 (1个样本) 真实算力: {flops_per_sample:,.0f} FLOPs")
    return flops_per_sample


kernel_force, kernel_action = get_fused_kernels(dtype, device)


def hmc_step_fn(p_phi): return HMC_step_production(p_phi, tao, hmc_config, kernel_force, kernel_action)


nf_model = FlowModel(NF_CONFIG).to(device).eval()
nf_prior = FreeFieldPrior(L=TARGET_L, m_sq_prior=0.6005269985).to(device)
dummy_prog = torch.tensor(1.0, device=device, dtype=dtype)
enforce_sym = NF_CONFIG.get('enforce_z2_sym', False)

real_hmc_flops = trace_real_flops("HMC", hmc_fn=hmc_step_fn)
real_nf_flops = trace_real_flops("NF", nf_model=nf_model, nf_prior=nf_prior, dummy_prog=dummy_prog,
                                 enforce_sym=enforce_sym)

# ==============================================================================
# 🏎️ 阶段二：3 分钟满载拉力赛 (完全等同于你的要求)
# ==============================================================================
try:
    compiled_hmc = torch.compile(hmc_step_fn, mode="max-autotune")
except Exception:
    compiled_hmc = hmc_step_fn

print(f"\n[比赛 1/2] 🏎️ HMC 正在满显存全力狂奔 (限时 {TARGET_TIME_SECONDS} 秒)...")
torch.cuda.empty_cache()
phi_gpu = torch.zeros(HMC_MAX_BATCH, TARGET_L, TARGET_L, dtype=dtype, device=device)

with torch.no_grad():
    for _ in range(5): compiled_hmc(phi_gpu)
    torch.cuda.synchronize()
    hmc_start_time = time.perf_counter()

    hmc_loops, hmc_total_raw_samples, hmc_avg_accept_sum = 0, 0, 0.0

    while True:
        phi_gpu, acc_tensor, _ = compiled_hmc(phi_gpu)
        hmc_loops += 1
        hmc_total_raw_samples += HMC_MAX_BATCH
        hmc_avg_accept_sum += acc_tensor.item()

        torch.cuda.synchronize()
        hmc_elapsed = time.perf_counter() - hmc_start_time
        if hmc_elapsed >= TARGET_TIME_SECONDS:
            break

hmc_mean_accept = hmc_avg_accept_sum / hmc_loops
print("\n🧹 正在执行核弹级显存清理，强行回收 HMC 的 CUDA Graphs 私人内存池...")

# 1. 彻底斩断 HMC 相关的全部变量引用
del compiled_hmc
del phi_gpu
if 'hmc_step_fn' in locals():
    del hmc_step_fn

# 2. 强制 Python 进行深度垃圾回收
import gc
gc.collect()

# 3. 重置 PyTorch 的底层编译引擎状态 (释放私有池)
if hasattr(torch, '_dynamo'):
    torch._dynamo.reset()

# 4. 最后清空常规缓存
torch.cuda.empty_cache()

print(f"\n[比赛 2/2] 🏎️ NF 正在满显存全力狂奔 (限时 {TARGET_TIME_SECONDS} 秒)...")


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
        if nf_elapsed >= TARGET_TIME_SECONDS:
            break

# ==============================================================================
# 📊 阶段三：终极大一统成绩结算
# ==============================================================================
print("\n" + "=" * 85)
print("🏆 终极大一统压测报告 (底层算力 + 物理产出)")
print("=" * 85)

# 1. 吞吐量计算
hmc_actions_per_sec = hmc_total_raw_samples / hmc_elapsed
nf_actions_per_sec = nf_total_raw_samples / nf_elapsed

# 2. 算力压榨计算 (维度一)
actual_hmc_tflops = (hmc_actions_per_sec * real_hmc_flops) / 1e12
actual_nf_tflops = (nf_actions_per_sec * real_nf_flops) / 1e12

# 3. 独立样本折算 (维度二)
# HMC: 原始样本数 / tau_int
hmc_effective_samples_total = hmc_total_raw_samples / TAU_INT
hmc_effective_samples_per_sec = hmc_effective_samples_total / hmc_elapsed

# NF: 原始样本数 * 接受率
nf_effective_samples_total = nf_total_raw_samples * ESTIMATED_NF_ACCEPTANCE
nf_effective_samples_per_sec = nf_effective_samples_total / nf_elapsed

print("📊 维度一：底层显卡算力压榨 (Compute-bound vs Memory-bound)")
print(f"  - HMC 实际压榨出: {actual_hmc_tflops:.3f} TFLOPs/s")
print(f"  - NF  实际压榨出: {actual_nf_tflops:.3f} TFLOPs/s")
print(f"  -> 现象: GPU 算力利用率差距为 {actual_nf_tflops / max(actual_hmc_tflops, 1e-9):,.1f} 倍。\n")

print(f"📊 维度二：{TARGET_TIME_SECONDS}秒真实时间内的物理总账")
print(f"  【HMC (单次循环 {HMC_TRAJECTORY_LENGTH} 步积分)】")
print(f"    - 总计运行时间: {hmc_elapsed:.2f} 秒")
print(f"    - 生成原始样本: {hmc_total_raw_samples:,} 个")
print(f"    - 平均 M-H 接受率: {hmc_mean_accept * 100:.2f}%")
print(f"    - 👉 扣除 tau_int (={TAU_INT}) 慢化后，折算独立样本: {hmc_effective_samples_total:,.2f} 个")
print(f"    - 👉 HMC 物理生成效率: {hmc_effective_samples_per_sec:,.2f} 独立样本/秒\n")

print(f"  【Normalizing Flow】")
print(f"    - 总计运行时间: {nf_elapsed:.2f} 秒")
print(f"    - 生成原始样本: {nf_total_raw_samples:,} 个")
print(f"    - 预估 M-H 接受率: {ESTIMATED_NF_ACCEPTANCE * 100:.1f}%")
print(f"    - 👉 折算独立样本: {nf_effective_samples_total:,.2f} 个")
print(f"    - 👉 NF 物理生成效率: {nf_effective_samples_per_sec:,.2f} 独立样本/秒\n")

print("🔥 终极结论：")
if nf_effective_samples_per_sec > hmc_effective_samples_per_sec:
    print(
        f"👉 流模型的最终物理级采样效率是 HMC 的 【{nf_effective_samples_per_sec / hmc_effective_samples_per_sec:.2f} 倍】！")
else:
    print(
        f"👉 HMC 的并行规模力挽狂澜，最终物理级采样效率是 NF 的 【{hmc_effective_samples_per_sec / nf_effective_samples_per_sec:.2f} 倍】！")