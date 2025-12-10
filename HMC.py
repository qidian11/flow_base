import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CONFIG = {
    'L': 3,
    'm2': -4.0,
    'lam': 5.113,
    'tao': 1.0,
    'leap_frog_step': 10,
    'save_steps': 10,
    'thermal_steps': 1000,
    'n_samples': 10000,
    'batch_size': 1,
}


def calculate_action(phi):
    # sum_x sum_miu phi_x*( 2*phi_x-phi_(x-miu)-phi_(x+miu) ) + m^2(phi_x)^2 + lam*phi_x^4
    phi_up = torch.roll(phi, shifts=-1, dims=2)
    phi_down = torch.roll(phi, shifts=1, dims=2)
    phi_right = torch.roll(phi, shifts=1, dims=3)
    phi_left = torch.roll(phi, shifts=-1, dims=3)
    kinetic_term = phi * phi * 2 * 2 - phi * phi_right - phi * phi_left - phi * phi_up - phi * phi_down
    potential_term = CONFIG['m2'] * phi * phi + CONFIG['lam'] * phi ** 4
    action = (kinetic_term + potential_term).sum(dim=(1, 2, 3))
    return action


def calculate_kinetic_energy(p):
    return 1 / 2 * torch.sum(p ** 2, dim=(1, 2, 3))


def calculate_hamiltonian(phi, p):
    action = calculate_action(phi)
    kinetic_energy = calculate_kinetic_energy(p)
    return kinetic_energy + action


def leap_frog(phi, p, tao):
    epsilon = tao / CONFIG['leap_frog_step']
    p_new = p + epsilon / 2 * get_force(phi)
    phi_new = phi.clone()
    for i in range(CONFIG['leap_frog_step']):
        phi_new = phi_new + epsilon * get_velocity(p_new)
        if i == CONFIG['leap_frog_step'] - 1:
            break
        p_new = p_new + epsilon * get_force(phi_new)
    p_new = p_new + epsilon / 2 * get_force(phi_new)
    return phi_new, p_new


def HMC_step(phi, p, tao):
    hamiltonian = calculate_hamiltonian(phi, p)
    phi_new, p_new = leap_frog(phi, p, tao)
    new_hamiltonian = calculate_hamiltonian(phi_new, p_new)
    delta_hamiltonian = new_hamiltonian - hamiltonian

    pass


def get_force(phi):
    return -(4 * phi + 2 * CONFIG['m2'] * phi + 4 * CONFIG['lam'] * phi ** 3)


def get_velocity(phi):
    return phi


def main():
    pass
