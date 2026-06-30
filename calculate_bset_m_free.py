import numpy as np
from scipy.optimize import root_scalar

CONFIG = [
    {'L': 6, 'm2': -4, 'lam': 6.975, 'tao': 1.62, 'leap_frog_step': 10, 'save_steps': 10, 'thermal_steps': 3000,
     'n_samples': 100000, 'bin_size': 100, 'bootstrap_time': 2000, 'batch_size': 128,
     'ensemble_save_path': 'HMC_6_.npz', 'loss_save_path': 'normalizing_flow_loss_6_20000.npy'},
    {'L': 8, 'm2': -4, 'lam': 6.008, 'tao': 1.5, 'leap_frog_step': 10, 'save_steps': 10, 'thermal_steps': 3000,
     'n_samples': 100000, 'bin_size': 100, 'bootstrap_time': 2000, 'batch_size': 128,
     'model_save_path': 'normalizing_flow_best_model_8_30000.pt',
     'loss_save_path': 'normalizing_flow_loss_8_30000.npy'},
    {'L': 10, 'm2': -4, 'lam': 5.550, 'tao': 1.39, 'leap_frog_step': 10, 'save_steps': 10, 'thermal_steps': 3000,
     'n_samples': 100000, 'bin_size': 100, 'bootstrap_time': 2000, 'batch_size': 128,
     'model_save_path': 'normalizing_flow_best_model_10_50000.pt',
     'loss_save_path': 'normalizing_flow_loss_10_50000.npy'},
    {'L': 12, 'm2': -4, 'lam': 5.276, 'tao': 1.28, 'leap_frog_step': 10, 'save_steps': 10, 'thermal_steps': 3000,
     'n_samples': 100000, 'bin_size': 100, 'bootstrap_time': 2000, 'batch_size': 128,
     'model_save_path': 'normalizing_flow_best_model_12_80000.pt',
     'loss_save_path': 'normalizing_flow_loss_12_80000.npy'},
    {'Type': 'HMC', 'L': 14, 'm2': -4, 'lam': 5.113, 'tao': 1.18, 'leap_frog_step': 10, 'save_steps': 10,
     'thermal_steps': 3000, 'n_samples': 100000, 'bin_size': 100, 'bootstrap_time': 2000, 'batch_size': 128,
     'double precision': True, 'model_save_path': 'normalizing_flow_best_model_14_100000.pt',
     'loss_save_path': 'normalizing_flow_loss_14_100000.npy', 'phi_ensemble_save_path': ''},
]


def calculate_m2_free(L, m2, lam, d=2):
    """
    通过最小化初始 KL 散度（求解 Gap Equation）获取最优 m_free^2
    """
    # 1. 构造动量空间的一维格点动能 K(k) = 2 - 2cos(p)
    k = np.arange(L)
    p = 2 * np.pi * k / L
    K_1d = 2 - 2 * np.cos(p)

    # 2. 推广到 d 维晶格 (默认 2D)
    K_grid = np.zeros((L,) * d)
    for i in range(d):
        shape = [1] * d
        shape[i] = L
        K_grid += K_1d.reshape(shape)

    V = L ** d

    # 3. 定义寻根目标函数 f(mu) = mu - m^2 - 6 * lambda * G_mu(0)
    # 当 f(mu) = 0 时即为最优解
    def objective(mu):
        if mu <= 0:  # 物理上 mu 必须大于0，避免零模式发散
            return -1e10
        # 计算 G_mu(0)
        G_0 = np.sum(1.0 / (2.0 * (K_grid + mu))) / V
        return mu - m2 - 6 * lam * G_0

    # 4. 求解方程
    # 随着 mu -> 0+，objective 趋于 -inf；随着 mu -> +inf，objective 趋于 +inf。
    # 必然在 (0, +inf) 之间存在唯一根。选用 Brent's method 进行安全求解。
    res = root_scalar(objective, bracket=[1e-6, 100.0], method='brentq')

    if res.converged:
        return res.root
    else:
        raise ValueError(f"Failed to converge for L={L}, m2={m2}, lam={lam}")


# 计算并更新 CONFIG
print(f"{'L':<5} | {'m2':<5} | {'lam':<7} | {'m_free^2':<10}")
print("-" * 35)

for conf in CONFIG:
    m2_free_opt = calculate_m2_free(conf['L'], conf['m2'], conf['lam'], d=2)
    conf['m2_free'] = m2_free_opt  # 将结果写回你的配置字典中
    print(f"{conf['L']:<5} | {conf['m2']:<5} | {conf['lam']:<7} | {m2_free_opt:<10.6f}")