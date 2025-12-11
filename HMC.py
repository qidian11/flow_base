import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from networkx.algorithms.tournament import hamiltonian_path

CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'tao': 0.6,
    'leap_frog_step': 10,
    'save_steps': 10,
    'thermal_steps': 1000,
    'n_samples': 1000,
    'batch_size': 64,
}


def calculate_action(phi):
    # sum_x sum_miu phi_x*( 2*phi_x-phi_(x-miu)-phi_(x+miu) ) + m^2(phi_x)^2 + lam*phi_x^4
    phi_up = torch.roll(phi, shifts=-1, dims=2)
    phi_down = torch.roll(phi, shifts=1, dims=2)
    phi_right = torch.roll(phi, shifts=1, dims=3)
    phi_left = torch.roll(phi, shifts=-1, dims=3)
    kinetic_term = 2 * phi * phi * 2  - phi * phi_right - phi * phi_left - phi * phi_up - phi * phi_down
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


def reverse_leapfrog(phi, p, tao):
    return leap_frog(phi, -p, tao)


def HMC_step(phi, p, tao, i):
    hamiltonian = calculate_hamiltonian(phi, p)
    phi_new, p_new = leap_frog(phi, p, tao)
    phi_reverse,p_reverse = reverse_leapfrog(phi_new, p_new, tao)
    reverse_hamiltonian = calculate_hamiltonian(phi_reverse, p_reverse)
    hamiltonian_ensemble.append(reverse_hamiltonian-hamiltonian)
    # print(phi_new.isnan().any())
    # nan_positions = torch.isnan(phi_new).nonzero()
    # print(nan_positions)
    new_hamiltonian = calculate_hamiltonian(phi_new, p_new)
    delta_hamiltonian = new_hamiltonian - hamiltonian
    prob = torch.exp(-delta_hamiltonian)
    rand_num = torch.rand(CONFIG['batch_size'])
    accept_mask = (rand_num < prob)
    if i % 100 == 0:
        print(f"第{i}步，accept ratio:{accept_mask.float().mean()*100}%")
    mask = accept_mask[:, None, None, None]
    phi_final = torch.where(mask, phi_new, phi)

    return phi_final


def get_force(phi):
    phi_up = torch.roll(phi, shifts=-1, dims=2)
    phi_down = torch.roll(phi, shifts=1, dims=2)
    phi_right = torch.roll(phi, shifts=1, dims=3)
    phi_left = torch.roll(phi, shifts=-1, dims=3)
    return -(8 * phi- 2 * (phi_up + phi_down + phi_right + phi_left)
             + 2 * CONFIG['m2'] * phi + 4 * CONFIG['lam'] * phi ** 3)


def get_velocity(p):
    return p



def get_2_point_correlation(ensemble,time_shift, space_shift):
    # ensemble.shape : samples/save_step, chains, 1, L, L
    ensemble_shifted = torch.roll(ensemble, shifts=(time_shift, space_shift), dims=(3,4))
    corr = ensemble_shifted * ensemble
    corr_grid_mean = corr.mean(dim=(2,3,4))
    corr_ensemble_mean = corr_grid_mean.mean(dim=1).mean(dim=0)
    return corr_ensemble_mean

def get_expected_phi(ensemble):
    expected_phi = ensemble.mean(dim=(2,3,4))
    expected_phi = expected_phi.mean(dim=1).mean(dim=0)
    return expected_phi

def get_connected_2_point_correlation(ensemble, time_shift, space_shift):
    corr_ensemble_mean = get_2_point_correlation(ensemble, time_shift, space_shift)
    expected_phi = get_expected_phi(ensemble)
    expected_phi_shifted = get_expected_phi(torch.roll(ensemble, shifts=(time_shift,space_shift), dims=(3,4)))
    return corr_ensemble_mean - expected_phi * expected_phi_shifted

def calculate_G_t_list(ensemble):
    G_t_list = []

    for t in range(CONFIG['L']):
        G_t_l_list = []
        for l in range(CONFIG['L']):
            conn = get_connected_2_point_correlation(ensemble, t, l)
            G_t_l_list.append(conn)
        G_t_mean = torch.stack(G_t_l_list).mean()
        G_t_list.append(G_t_mean)

    return torch.stack(G_t_list)


hamiltonian_torch = torch.randn(CONFIG['batch_size'])
hamiltonian_ensemble = []

def main():
    phi = torch.randn(CONFIG['batch_size'], 1, CONFIG['L'], CONFIG['L'],dtype=torch.float64)
    print(phi.isnan().any())
    ensemble = []

    for i in range(CONFIG['thermal_steps']):
        p = torch.randn(CONFIG['batch_size'], 1, CONFIG['L'], CONFIG['L'],dtype=torch.float64)
        phi_new = HMC_step(phi, p, CONFIG['tao'], i)
        # if phi.isnan().any(): continue
        phi = phi_new
    for i in range(CONFIG['n_samples']):
        p = torch.randn(CONFIG['batch_size'], 1, CONFIG['L'], CONFIG['L'],dtype=torch.float64)
        phi_new = HMC_step(phi, p, CONFIG['tao'], i)
        # if phi.isnan().any(): continue
        phi = phi_new
        if i % CONFIG['save_steps'] == 0:
            ensemble.append(phi.clone().detach())
    ensemble_tensor = torch.stack(ensemble, dim=0)
    hamiltonian_ensemble_tensor = torch.cat(hamiltonian_ensemble).reshape(-1)
    print(hamiltonian_ensemble_tensor.shape)
    plt.plot(hamiltonian_ensemble_tensor.numpy())
    plt.show()
    G_t = calculate_G_t_list(ensemble_tensor)
    # G_t = compute_zero_momentum_Gt(ensemble_tensor)
    plt.plot(G_t.numpy(),marker='o')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('G_t')
    plt.title('2-point fuction of 2-dim lattice φto4 theory')
    plt.show()

def calculate_effective_mass(ensemble):
    pass



if __name__ == '__main__':
    main()
