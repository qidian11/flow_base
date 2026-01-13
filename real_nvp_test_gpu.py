import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data  # 引入数据处理模块
import time
import matplotlib.pyplot as plt

# --- 配置 ---
BATCH_SIZE = 2048  # 关键参数：一批处理多少个数据
# 如果显卡强（如 B580），建议设大点（1024, 2048, 4096）以跑满显卡
# 如果设太小（如 32），显卡大部分时间在等 Python 循环

device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
print(f"Running on: {device}")

# --- 1. 准备数据 ---
# 生成 20 万个数据
x = torch.linspace(-100, 100, 200000).unsqueeze(1)
y = x.pow(2)

# 将数据封装成 Dataset
torch_dataset = Data.TensorDataset(x, y)

# 创建 DataLoader (加载器)
# 它会帮我们自动要把数据打乱 (shuffle)，并切分成一个个 Batch
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 每次训练打乱顺序，有助于训练
    num_workers=0,  # Windows 下通常设为 0，Linux 可以设为 4
)


# --- 2. 定义网络 ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 128)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.layer2 = nn.Linear(128, 128)  # 稍微加宽一点
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.layer3(x)
        return x


net = SimpleNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# --- 3. 训练循环 (Epoch + Batch) ---
print(f"Start Training with Batch Size {BATCH_SIZE}...")
start_time = time.time()

for epoch in range(50):  # 只需要较少的 Epoch，因为每个 Epoch 内部更新了很多次

    # 这一步会自动遍历每一个 Batch
    for step, (batch_x, batch_y) in enumerate(loader):
        # 记得把每个 Batch 的数据搬运到 GPU/XPU
        b_x = batch_x.to(device)
        b_y = batch_y.to(device)

        prediction = net(b_x)
        loss = loss_func(prediction, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch: {epoch}, Last Batch Loss: {loss.item():.4f}")

# 计时结束
if device.type == 'xpu': torch.xpu.synchronize()
end_time = time.time()

print(f"Total Time: {end_time - start_time:.2f}s")

# --- 4. 绘图验证 ---
# 测试时可以用全量数据（如果显存够大），或者取一部分
x_test = torch.linspace(-150, 150, 500).unsqueeze(1).to(device)
y_test = x_test.pow(2)
with torch.no_grad():
    pred_test = net(x_test)

plt.plot(x_test.cpu().numpy(), y_test.cpu().numpy(), 'r-', label='Real')
plt.plot(x_test.cpu().numpy(), pred_test.cpu().numpy(), 'b--', label='Predict')
plt.legend()
plt.show()