import numpy as np
import matplotlib.pyplot as plt
from handwritten_gradient_descent import *


# 假设你的所有类定义都在这里，或者你从原来的文件中 import 进来
# from your_library import Variable, Linear, NN, Adam, ReLU, Pow, etc.

# ==========================================
# 1. 补充工具类 (适配你的框架)
# ==========================================

class ReLULayer:
    """
    为了适配 NN 类中 layer.forward(x) 的调用方式，
    我们需要把 Function 包装成一个 Layer。
    """

    def forward(self, x):
        # 调用 Variable 中定义的 .relu() 快捷方式
        return x.relu()

    # ==========================================
class LeakyReLULayer:
    def forward(self, x):
        return x.leaky_relu()

# 2. 准备数据
# ==========================================

# 目标函数: y = 3x^2 + 2x + 1
def target_function(x):
    return 3 * x ** 2 + 2 * x + 1


np.random.seed(42)
# 生成 -1 到 1 之间的 100 个随机点
X_raw = np.random.uniform(-1, 1, (100, 1))
# 生成标签并加入一点点噪声
Y_raw = target_function(X_raw) + np.random.normal(0, 0.05, (100, 1))

# 转为 Variable (虽然 Variable 默认处理 numpy，但显式转换是个好习惯)
# 注意：输入数据不需要梯度，所以默认 store_history=False 即可
inputs = Variable(X_raw)
targets = Variable(Y_raw)

# ==========================================
# 3. 构建模型
# ==========================================

# 搭建 3 层神经网络: 1 -> 16 -> 16 -> 1
# 使用 Xavier 初始化已经在 Linear 中实现
layer1 = Linear(1, 16)
layer2 = Linear(16, 16)
layer3 = Linear(16, 1)

# 组装网络
model = NN([
    layer1,
    LeakyReLULayer(),
    layer2,
    LeakyReLULayer(),
    layer3
])

# ==========================================
# 4. 优化器设置
# ==========================================

# 将模型参数传给 Adam
optimizer = Adam(model.parameters, lr=0.05)

# ==========================================
# 5. 训练循环
# ==========================================

losses = []
epochs = 2000

print(f"开始训练，拟合 y = 3x^2 + 2x + 1 ...")

for epoch in range(epochs):

    # --- 前向传播 ---
    y_pred = model.forward(inputs)

    # --- 计算 MSE Loss ---
    # 公式: mean((y_pred - targets) ^ 2)
    # 由于 Variable 尚未实现 __sub__，我们用 y_pred + (targets * -1) 代替
    diff = y_pred + (targets * -1.0)

    # 由于 Variable 尚未实现 __pow__，我们直接调用 Pow.apply
    loss = Pow.apply(diff, 2).mean()

    # 记录 Loss 用于画图
    losses.append(loss.data)

    # --- 反向传播 ---
    # 1. 清空旧梯度
    model.clear_gradient()

    # 2. 开启 Loss 的梯度记录 (如果是 Variable 内部创建的节点，通常不需要手动开启，
    #    但为了保险起见，确保根节点能反向传播)
    #    注：你的实现中 backward 会自动处理，这里直接调用即可。
    loss.backward()

    # --- 参数更新 ---
    optimizer.optimize()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.data:.6f}")

print(f"最终 Loss: {loss.data:.6f}")

# ==========================================
# 6. 验证与可视化
# ==========================================

# 生成平滑曲线用于对比
X_test = np.linspace(-1, 1, 100).reshape(-1, 1)
Y_test_true = target_function(X_test)
Y_test_pred = model.forward(Variable(X_test)).data

# 打印几个预测值对比
print("\n--- 预测结果采样 ---")
for i in range(0, 100, 20):
    x_val = X_test[i][0]
    true_val = Y_test_true[i][0]
    pred_val = Y_test_pred[i][0]
    print(f"x={x_val: .2f} | 真实值={true_val:.3f} | 预测值={pred_val:.3f}")

# 绘图
plt.figure(figsize=(12, 5))

# 子图1: 训练 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")  # 使用对数坐标看下降趋势

# 子图2: 拟合效果
plt.subplot(1, 2, 2)
plt.scatter(X_raw, Y_raw, color='gray', alpha=0.5, label='Training Data (Noisy)')
plt.plot(X_test, Y_test_true, color='blue', linestyle='--', label='True Function')
plt.plot(X_test, Y_test_pred, color='red', linewidth=2, label='NN Prediction')
plt.title("Quadratic Fitting Result")
plt.legend()

plt.tight_layout()
plt.show()