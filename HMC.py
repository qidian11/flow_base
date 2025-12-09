import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fftpack import shift
from sympy.physics.mechanics import kinetic_energy

CONFIG = {
    'L': 3,
    'm2':-4.0,
    'lam': 5.113,
    'tao': 1.0,
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
    kinetic_term = phi*phi*2*2-phi*phi_right-phi*phi_left-phi*phi_up-phi*phi_down
    potential_term = CONFIG['m2']*phi*phi + CONFIG['lam']*phi**4
    action = (kinetic_term + potential_term).sum(dim=(1,2,3))
    return action


def main():
    pass