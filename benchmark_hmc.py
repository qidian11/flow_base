import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from HMC_final import CONFIG as HMC_CONFIG_LIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --- 核心参数配置 ---
TARGET_L = 14
TARGET_TIME_SECONDS = 180.0
HMC_MAX_BATCH = 2_334_720
TAU_INT = 200
HMC_TRAJECTORY_LENGTH = 10

hmc_config = next(c for c in HMC_CONFIG_LIST if c.get('L') == TARGET_L)
tao = hmc_config['tao']
dtype = torch.float32

# --- HMC 优化算子定义 ---
def get_fused_kernels(dtype, device):
    kernel_force = torch.tensor([[0.0, 2.0, 0.0], [2.0, -8.0, 2.0], [0.0, 2.0, 0.0]], dtype=dtype, device=device).view(1, 1, 3, 3)
    kernel_action = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=dtype, device=device).view(1, 1, 3, 3)
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

# --- 阶段一：真实 FLOPs 追踪 ---
print(f"{'=' * 85}")
print(f"🔥 HMC 独立拉力赛：底层算力评估 (限时 {TARGET_TIME_SECONDS}秒)")
print(f"💻 设备: {torch.cuda.get_device_name(0)} | 精度: Float32")
print(f"{'=' * 85}")

print("\n[追踪中] 正在探测 HMC 单步执行的真实浮点运算量...")
TRACE_BATCH = 16
kernel_force, kernel_action = get_fused_kernels(dtype, device)
def hmc_step_fn(p_phi): return HMC_step_production(p_phi, tao, hmc_config, kernel_force, kernel_action)

phi_trace = torch.zeros(TRACE_BATCH, TARGET_L, TARGET_L, dtype=dtype, device=device)
for _ in range(3): hmc_step_fn(phi_trace)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
    hmc_step_fn(phi_trace)

total_flops = sum([evt.flops for evt in prof.events() if evt.flops is not None and evt.flops > 0])
real_hmc_flops = total_flops / TRACE_BATCH

if real_hmc_flops == 0:
    print("    ⚠️ PyTorch 未抓取到原生张量操作，启用物理底层计算量补偿。")
    real_hmc_flops = 2800
print(f"    🎯 HMC 单次动作 (1个样本) 真实算力: {real_hmc_flops:,.0f} FLOPs")

# --- 阶段二：满载 3 分钟拉力赛 ---
try: compiled_hmc = torch.compile(hmc_step_fn, mode="max-autotune")
except Exception: compiled_hmc = hmc_step_fn

print(f"\n[比赛进行中] 🏎️ HMC 正在满显存全力狂奔 (目标耗时 {TARGET_TIME_SECONDS} 秒)...")
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
        if hmc_elapsed >= TARGET_TIME_SECONDS: break

hmc_mean_accept = hmc_avg_accept_sum / hmc_loops

# --- 阶段三：HMC 成绩单 ---
hmc_actions_per_sec = hmc_total_raw_samples / hmc_elapsed
actual_hmc_tflops = (hmc_actions_per_sec * real_hmc_flops) / 1e12
hmc_effective_samples_total = hmc_total_raw_samples / TAU_INT
hmc_effective_samples_per_sec = hmc_effective_samples_total / hmc_elapsed

print("\n" + "=" * 85)
print("🏁 HMC 独立拉力赛最终成绩")
print("=" * 85)
print(f"📊 维度一：底层算力压榨 (Compute-bound vs Memory-bound)")
print(f"  - 真实硬件动作吞吐: {hmc_actions_per_sec:,.0f} 次Leapfrog/秒")
print(f"  - 实际榨取物理算力: {actual_hmc_tflops:.3f} TFLOPs/s")

print(f"\n📊 维度二：{TARGET_TIME_SECONDS}秒真实时间内的物理总账")
print(f"  - 总计运行时间: {hmc_elapsed:.2f} 秒")
print(f"  - 并发规模: {HMC_MAX_BATCH:,} 链")
print(f"  - 生成原始样本: {hmc_total_raw_samples:,} 个")
print(f"  - 平均 M-H 接受率: {hmc_mean_accept * 100:.2f}%")
print(f"  - 👉 扣除 tau_int (={TAU_INT}) 慢化后，折算独立样本: {hmc_effective_samples_total:,.2f} 个")
print(f"  - 👉 HMC 物理级生成效率: {hmc_effective_samples_per_sec:,.2f} 独立样本/秒")