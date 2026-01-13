import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import time

# ================= 配置区域 =================
# 1. 修改这里切换设备: "xpu" 或 "cpu"
TARGET_DEVICE = "xpu"

# 2. 统一参数 (保证工作量完全一致)
DATA_SIZE = 100000  # 数据量：10万
BATCH_SIZE = 1024  # 批大小：1024
EPOCHS = 1000  # 轮数：只跑3轮，不然CPU要跑太久


# ===========================================

def get_device(prefer="auto"):
    if prefer == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if prefer == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
    if prefer == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
        if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


device = get_device(prefer=TARGET_DEVICE)
print(f"🔥 当前测试设备: {device}")
print(f"📦 数据总量: {DATA_SIZE}, Batch Size: {BATCH_SIZE}")

# --- 1. 准备数据 ---
# 保证 CPU 和 GPU 处理的数据一模一样
x = torch.linspace(-100, 100, DATA_SIZE).unsqueeze(1)
y = x.pow(2)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


# --- 2. 定义网络 (增加一点计算复杂度) ---
class BenchNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 稍微加宽一点，让计算量更足，减少纯粹的通信开销干扰
        self.net = nn.Sequential(
            nn.Linear(1, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)


net = BenchNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

# --- 3. 预热 (Warm Up) ---
# 为了去除第一次加载模型和编译的开销，先空跑几次
print("正在预热设备...")
dummy_input = torch.randn(BATCH_SIZE, 1).to(device)
for _ in range(5):
    net(dummy_input)
if device.type == 'xpu': torch.xpu.synchronize()

# --- 4. 正式测试 ---
print(f"🚀 开始测试 (共 {EPOCHS} 轮)...")
total_start = time.time()
total_samples_processed = 0

for epoch in range(EPOCHS):
    epoch_start = time.time()

    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = batch_x.to(device)
        b_y = batch_y.to(device)

        prediction = net(b_x)
        loss = loss_func(prediction, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_samples_processed += b_x.size(0)

    # 确保当前 Epoch GPU 跑完了
    if device.type == 'xpu': torch.xpu.synchronize()
    epoch_end = time.time()
    print(f"Epoch {epoch + 1}/{EPOCHS} 耗时: {epoch_end - epoch_start:.4f}秒")

total_end = time.time()
total_duration = total_end - total_start

# --- 5. 最终算分 ---
throughput = total_samples_processed / total_duration

print("\n" + "=" * 40)
print(f"🏁 测试结果 - {device}")
print(f"总耗时:     {total_duration:.4f} 秒")
print(f"吞吐量:     {throughput:.0f} samples/second (越大越好)")
print("=" * 40)