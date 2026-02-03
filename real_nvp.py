import numpy as np
from sympy.physics.mechanics import kinetic_energy

from handwritten_gradient_descent import *

CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'tao': 1.18,
    'leap_frog_step': 10,
    'save_steps': 10,
    'thermal_steps': 1000,
    'n_samples': 120000,
    'bin_size': 100,
    'bootstrap_time': 2000,
    'batch_size': 1024,
}


def get_action(phi):
    kinetic_term = get_kinetic_term_for_action(phi)
    mass_term = get_mass_term_for_action(phi)
    interaction_term = get_interaction_term_for_action(phi)
    return AddN.apply(kinetic_term,mass_term,interaction_term).sum()

def get_kinetic_term_for_action(phi):
    # phi_up = np.roll(phi, -1, 0)
    # phi_down = np.roll(phi, 1, 0)
    # phi_left = np.roll(phi, -1, 1)
    # phi_right = np.roll(phi, 1, 1)
    phi_up = phi.roll(-1, 0)
    phi_down = phi.roll(1, 0)
    phi_left = phi.roll(-1, 1)
    phi_right = phi.roll(1, 1)
    shift_term = AddN.apply(phi_up, phi_down, phi_left, phi_right)
    return 4*Pow.apply(phi,2)+phi*shift_term


def get_mass_term_for_action(phi):
    return CONFIG["m2"] * Pow.apply(phi,2)

def get_interaction_term_for_action(phi):
    return CONFIG["lam"] * Pow.apply(phi,4)


def create_mask(n):
    L = CONFIG['L']
    indices = np.arange(L)
    mask_2d = (indices[:, None] + indices[None, :]) % 2 == 0
    mask_flat = mask_2d.flatten()
    mask_batch = np.broadcast_to(mask_flat, (n, L*L))
    return mask_batch


def get_log_jacobian(phi, layers):
    # 设置为第一层偶数格子不变
    if len(layers) % 2 == 0:
        mask = create_mask()
    else:
        mask = create_mask()
        mask = ~mask
    total_log_jacobian = 0
    for layer in layers.reverse():
        s_phi = layer.forward(phi[mask])
        exp_s_phi = Exp.apply(s_phi)
        jacobian = exp_s_phi.prod()
        log_jacobian = jacobian.log()
        total_log_jacobian += log_jacobian
    return total_log_jacobian

def mask_select(x, mask):
    return MaskSelect.apply(x, mask)

def combine(z_a, z_b, mask):
    return Combine.apply(z_a, z_b, mask)

def mask_select_for_batch(x, mask):
    return MaskSelectForBatch.apply(x, mask)

def combine_for_batch(z_a, z_b, mask):
    return CombineForBatch.apply(z_a, z_b, mask)


def compute_log_prior(z):
    # It can be ignored for gradients
    const = np.log(2 * np.pi)
    return -0.5 * (z ** 2 + const).sum()


class PhiToZNVP(NN):
    def set_all_parameters(self):
        self.params = []  # 确保这里使用你刚才修改后的变量名 (self.params)
        for layer in self.layers:
            s_net, t_net = layer
            # 递归收集 s_net 和 t_net 的参数
            self.params += s_net.parameters()
            self.params += t_net.parameters()

    def compute_loss(self, z, log_total_jacobian):
        batch_size = z.data.shape[0]
        log_r_z = (-0.5*(z**2)).sum()
        loss = (-1 * log_r_z - log_total_jacobian) / batch_size
        return loss

    def forward(self, phi):
        # 获取原始形状，以便后续如果需要还原
        original_shape = phi.data.shape

        # 将 phi 从 (batch_size, L, L) 展平为 (batch_size, L*L)
        # 如果 phi 已经是 (batch_size, split_dim)，此操作依然安全
        batch_size = original_shape[0]
        flattened_data = phi.data.reshape(batch_size, -1)
        phi_flat = Variable(flattened_data)
        z = phi_flat
        base_mask = create_mask(phi.data.shape[0])
        log_total_jacobian = 0
        for i,layer in enumerate(self.layers):
            if i % 2 == 0:
                mask = base_mask
            else:
                mask = ~base_mask

            z_a = mask_select_for_batch(z, mask)
            z_b = mask_select_for_batch(z, ~mask)
            s, t = layer
            s_out = s.forward(z_a)
            t_out = t.forward(z_a)

            z_b = Exp.apply(s_out) * z_b + t_out

            log_total_jacobian += s_out.sum()
            z = combine_for_batch(z_a, z_b, mask)
        loss = self.compute_loss(z, log_total_jacobian)
        self.loss = loss
        return self.loss

L = CONFIG['L']
split_dim = L*L // 2
hidden_dim = 128
coupling_layer_num = 8

def create_st_network(input_dim, hidden_dim):
    return NN([
        Layer(Linear(input_dim, hidden_dim), LeakyReLU()),
        Layer(Linear(hidden_dim,hidden_dim), LeakyReLU()),
        Linear(hidden_dim, input_dim)
    ])

coupling_layers = []
for i in range(coupling_layer_num):
    s_net = create_st_network(split_dim, hidden_dim)
    t_net = create_st_network(split_dim, hidden_dim)
    coupling_layers.append((s_net, t_net))

phi_to_znvp = PhiToZNVP(coupling_layers)

filename = "configs_L14_N12800.npy"

try:
    # 2. 加载数据
    phi_hmc_all = np.load(filename)
    phi_hmc_all = phi_hmc_all.reshape(phi_hmc_all.shape[0], L, L)

    print(f"✅ File read successfully: {filename}")
    print(f"Data Shape: {phi_hmc_all.shape}")


    print(f"Data Type(Dtype): {phi_hmc_all.dtype}")


except FileNotFoundError:
    print(f"❌ File not found: {filename}，please check the path.。")


optimizer = Adam(phi_to_znvp.parameters(),lr=1e-4)
N_SAMPLES = phi_hmc_all.shape[0]
BATCH_SIZE = CONFIG['batch_size']
epochs = 1000

for epoch in range(epochs):

    # 1. 【关键步骤】生成打乱的索引
    # np.random.permutation(N) 会生成一个 0 到 N-1 的随机排列数组
    # 例如 N=5 -> [3, 0, 4, 1, 2]
    shuffled_indices = np.random.permutation(N_SAMPLES)

    # 2. 按 Batch_Size 步长遍历
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        current_indices = shuffled_indices[i: i + BATCH_SIZE]

        phi_batch_np = phi_hmc_all[current_indices]

        phi_batch = Variable(phi_batch_np)

        loss = phi_to_znvp.forward(phi_batch)

        phi_to_znvp.clear_gradient()
        phi_to_znvp.clear_gradient()
        phi_to_znvp.backward()
        optimizer.optimize()

    print(f"Loos:{loss.data}")
    # print(f"len(optimizer.parameters):{len(optimizer.parameters)}")
    print(f"Epoch {epoch} finished.")