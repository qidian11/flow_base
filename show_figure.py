import numpy as np
import matplotlib.pyplot as plt

steps = 20000
my_loss_data = np.load("cnn_res_model_loss_history_coupling_layers_16_hidden_layers_6_hidden_channels_16_iterations_20000.npy")
print(my_loss_data[-100:])
def moving_average(data, window_size=100):
    """计算滑动平均以平滑曲线"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 1. 设置画布大小
plt.figure(figsize=(10, 6))

# 2. 绘制原始 Loss 曲线 (使用较低的透明度 alpha，避免遮挡主要趋势)
plt.plot(my_loss_data, alpha=0.3, color='gray', label='Original Loss')


plt.plot(my_loss_data, color='#1f77b4', linewidth=1, label='Training Loss')

# 4. 添加标题和标签
plt.title('Training Loss History', fontsize=16)
plt.xlabel('Training Steps (or Epochs)', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)

plt.yscale('symlog')
# 5. 显示图例和网格
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 6. 显示图像 (如果需要保存图像，可以使用 plt.savefig("loss_curve.png"))
plt.show()