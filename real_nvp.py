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
    'batch_size': 64,
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


def create_mask():
    L = CONFIG['L']
    indices = np.arange(L)
    mask = (indices[:, None] + indices[None, :]) % 2 == 0
    return mask

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

def compute_log_prior(z):
    # It can be ignored for gradients
    const = np.log(2 * np.pi)
    return -0.5 * (z ** 2 + const).sum()


class Layer:
    def __init__(self, linear, leaky_relu):
        self.linear = linear
        self.leaky_relu = leaky_relu
        self.parameters = linear.parameters()

    def forward(self, phi):
        z = self.linear.forward(phi)
        z = self.leaky_relu(z)
        return z


class NVP(NN):
    def forward(self, z):
        base_mask = create_mask()
        for i,layer in enumerate(self.layers):
            mask = ~base_mask
            z_a = mask_select(z, mask)
            z_b = mask_select(z, ~mask)
            s, t = layer
            z_b = Exp.apply(-1*s.forward(z_a)) * (z_b-t.forward(z_a))
        return z_a,z_b





