import math
import random
import csv

# --- 配置 ---
CONFIG = {
    'L': 6,  # 6x6 格点
    'm2': -4.0,
    'lam': 5.113,
    'tao': 1.1,
    'leap_frog_step': 10,
    'n_samples': 2,  # 只跑1个样本足够看爆炸过程了
    'thermal_steps': 0  # 不需要热化，直接开始追踪
}


# --- 辅助函数 ---

def create_grid(size):
    return [0.0] * (size * size)  # 使用一维数组模拟二维，方便索引


def random_normal_grid(size):
    return [random.gauss(0, 1) for _ in range(size * size)]


def get_idx(i, j):
    """处理周期性边界条件，将(i,j)映射到一维索引"""
    L = CONFIG['L']
    return (i % L) * L + (j % L)


def calculate_action(phi):
    L = CONFIG['L']
    action = 0.0
    for i in range(L):
        for j in range(L):
            idx = get_idx(i, j)
            val = phi[idx]

            # 邻居: 上下左右
            sum_neighbors = (phi[get_idx(i - 1, j)] +
                             phi[get_idx(i + 1, j)] +
                             phi[get_idx(i, j - 1)] +
                             phi[get_idx(i, j + 1)])

            # 动能部分: 2 * phi * (2*phi - sum_neighbors)
            # 注意：原公式 kinetic = 2 * phi * (2*phi - sum_neighbors)
            # 这里的系数 2 取决于离散化定义，保持和你原始逻辑一致
            # 原代码: 2*phi*phi*2 - phi*neighbors
            # = 4*phi^2 - phi*neighbors
            kinetic = 4.0 * val * val - val * sum_neighbors
            potential = CONFIG['m2'] * val * val + CONFIG['lam'] * (val ** 4)
            action += kinetic + potential
    return action


def calculate_kinetic_energy(p):
    return 0.5 * sum(x * x for x in p)


def calculate_hamiltonian(phi, p):
    try:
        return calculate_action(phi) + calculate_kinetic_energy(p)
    except OverflowError:
        return float('inf')


# --- 核心物理计算与记录 ---

def get_force(phi, writer, step_info):
    """计算力，并直接记录到CSV"""
    L = CONFIG['L']
    force = [0.0] * (L * L)

    try:
        for i in range(L):
            for j in range(L):
                idx = get_idx(i, j)
                val = phi[idx]

                sum_neighbors = (phi[get_idx(i - 1, j)] +
                                 phi[get_idx(i + 1, j)] +
                                 phi[get_idx(i, j - 1)] +
                                 phi[get_idx(i, j + 1)])

                # Force = - dS/dphi
                # dS/dphi = 8*phi - 2*sum_neighbors + 2*m2*phi + 4*lam*phi^3
                d_action = (8 * val - 2 * sum_neighbors) + \
                           (2 * CONFIG['m2'] * val + 4 * CONFIG['lam'] * (val ** 3))

                force[idx] = -d_action

        # 记录 Force
        log_row(writer, step_info, "Force", force, 0)  # Force不贡献哈密顿量，填0
        return force

    except OverflowError:
        log_row(writer, step_info, "Force_EXPLODED", force, 0)
        return force  # 返回已计算的部分


def leap_frog(phi, p, tao, writer, base_info):
    """
    base_info: {'step': i, 'phase': 'Forward'/'Reverse'}
    """
    L = CONFIG['L']
    epsilon = tao / CONFIG['leap_frog_step']

    # 复制列表，避免修改原始引用
    phi_curr = list(phi)
    p_curr = list(p)

    # 记录初始状态
    base_info['lf_step'] = 'Start'
    H = calculate_hamiltonian(phi_curr, p_curr)
    log_row(writer, base_info, "Phi", phi_curr, H)
    log_row(writer, base_info, "P", p_curr, H)

    # === 1. Half step for Momentum ===
    base_info['lf_step'] = 0
    # 计算力 (内部已记录 Force 到 CSV)
    force = get_force(phi_curr, writer, base_info)

    # 更新 P
    for k in range(L * L):
        p_curr[k] += (epsilon / 2.0) * force[k]

    H = calculate_hamiltonian(phi_curr, p_curr)
    log_row(writer, base_info, "P_Half_Update", p_curr, H)

    # === 2. Loop ===
    for i in range(CONFIG['leap_frog_step']):
        base_info['lf_step'] = i + 1

        # Update Position (Phi)
        for k in range(L * L):
            phi_curr[k] += epsilon * p_curr[k]

        H = calculate_hamiltonian(phi_curr, p_curr)
        log_row(writer, base_info, "Phi_Update", phi_curr, H)

        # Update Momentum (Full step, except last)
        if i != CONFIG['leap_frog_step'] - 1:
            force = get_force(phi_curr, writer, base_info)
            for k in range(L * L):
                p_curr[k] += epsilon * force[k]

            H = calculate_hamiltonian(phi_curr, p_curr)
            log_row(writer, base_info, "P_Full_Update", p_curr, H)

    # === 3. Final Half step for Momentum ===
    base_info['lf_step'] = 'Final'
    force = get_force(phi_curr, writer, base_info)
    for k in range(L * L):
        p_curr[k] += (epsilon / 2.0) * force[k]

    H = calculate_hamiltonian(phi_curr, p_curr)
    log_row(writer, base_info, "P_Final_Update", p_curr, H)

    return phi_curr, p_curr


def log_row(writer, info, var_type, grid_data, H):
    """格式化并写入一行 CSV"""
    # 格式化数据，保留6位小数，如果是 nan/inf 则直接转字符串
    formatted_grid = []
    for x in grid_data:
        if math.isnan(x) or math.isinf(x):
            formatted_grid.append(str(x))
        else:
            formatted_grid.append(f"{x:.6f}")

    row = [
              info['step'],
              info['phase'],
              info['lf_step'],
              var_type,
              f"{H:.6f}" if not (math.isnan(H) or math.isinf(H)) else str(H)
          ] + formatted_grid
    writer.writerow(row)


def main():
    # 初始化
    L = CONFIG['L']
    phi = random_normal_grid(L)

    filename = 'hmc_trace_log.csv'
    print(f"Start tracing. Output will be written to {filename}")

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # 生成表头: Grid_00, Grid_01 ... Grid_55
        grid_headers = [f"Grid_{i}" for i in range(L * L)]
        header = ["Step", "Phase", "LF_SubStep", "Var_Type", "Hamiltonian"] + grid_headers
        writer.writerow(header)

        # 运行采样
        total_steps = CONFIG['n_samples']
        for step in range(total_steps):
            print(f"Running Step {step}...")

            # 生成动量
            p = random_normal_grid(L)

            # 记录初始状态
            base_info = {'step': step, 'phase': 'Forward', 'lf_step': 'Init'}

            # --- 正向 Leapfrog ---
            phi_new, p_new = leap_frog(phi, p, CONFIG['tao'], writer, base_info)

            # --- 反向 Leapfrog (验证用) ---
            # 反转动量
            p_neg = [-x for x in p_new]
            base_info['phase'] = 'Reverse'

            # 记录反转后的初始态
            log_row(writer, base_info, "P_Reversed", p_neg, calculate_hamiltonian(phi_new, p_neg))

            phi_rev, p_rev = leap_frog(phi_new, p_neg, CONFIG['tao'], writer, base_info)

            # Metropolis Accept/Reject (为了完整性，虽然主要目的是看 Leapfrog 内部)
            H_start = calculate_hamiltonian(phi, p)
            H_new = calculate_hamiltonian(phi_new, p_new)
            delta_H = H_new - H_start

            try:
                prob = math.exp(-delta_H)
            except OverflowError:
                prob = 0.0

            if random.random() < prob:
                phi = phi_new
                print(f"Step {step}: Accepted")
            else:
                print(f"Step {step}: Rejected")

    print("Done. Check hmc_trace_log.csv")


if __name__ == '__main__':
    main()