import numpy as np
import matplotlib.pyplot as plt

# 根据 test_kinetic_term.py 的命名规则，生成的文件名如下：
file_A = 'observables_Exp_A_Pure_Kinetic_dp_False_L14_c10_iter_15000.npz'
file_B = 'observables_Exp_B_Pure_Potential_dp_False_L14_c10_iter_15000.npz'

try:
    # 加载数据
    data_A = np.load(file_A)
    data_B = np.load(file_B)

    # 提取 Iteration 和 接受率 (Acc)
    steps_A, acc_A = data_A['steps'], data_A['acc']
    steps_B, acc_B = data_B['steps'], data_B['acc']

    # 开始绘图
    plt.figure(figsize=(10, 6), dpi=150)

    # 绘制实验 A
    plt.plot(steps_A, acc_A * 100, marker='o', linestyle='-', linewidth=2,
             label='Exp A: Kinetic action', color='#1f77b4')

    # 绘制实验 B
    plt.plot(steps_B, acc_B * 100, marker='s', linestyle='--', linewidth=2,
             label='Exp B: Potential action', color='#d62728')

    # 细节设置
    plt.title('MCMC Acceptance Rate Comparison', fontsize=15, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Acceptance Rate (%)', fontsize=12)

    # 设置 y 轴范围，由于最高是 87.67%，范围设在 0-100 比较合适
    plt.ylim(0, 100)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig('mcmc_acceptance_rate_comparison.png', dpi=300)
    plt.show()

    print("✅ 绘图完成！图表已保存为 mcmc_acceptance_rate_comparison.png")

except FileNotFoundError as e:
    print(f"❌ 找不到数据文件，请检查当前目录下是否存在对应的 .npz 文件。\n报错信息: {e}")