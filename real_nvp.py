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
    'batch_size': 128,
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


