import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import math
import copy

# ---------------------------
# CONFIG (adjust for debug)
# ---------------------------
CONFIG = {
    'L': 14,
    'm2': -4.0,
    'lam': 5.113,
    'tao': 0.1,
    'leap_frog_step': 10,
    'save_steps': 10,
    'thermal_steps': 200,   # debug: keep small; change to 1000 later
    'n_samples': 5000,       # debug: keep small; change to 10000 later
    'batch_size': 64,
    'diagnostic_threshold': 1e6,  # threshold for "too large"
    'history_len': 10,
}

# ---------------------------
# Utilities
# ---------------------------
def roll(x, shift, axis):
    """Wrapper for np.roll, kept for readability."""
    return np.roll(x, shift=shift, axis=axis)

def stats_of_array(a):
    """Return basic stats for diagnostics (as floats)."""
    return {
        'max_abs': float(np.max(np.abs(a))),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'mean': float(np.mean(a)),
        'nan': bool(np.isnan(a).any()),
        'inf': bool(np.isinf(a).any())
    }

# ---------------------------
# Global history buffers
# ---------------------------
# Each entry holds stats for K, M, L (the splitted force contributions)
force_history = deque(maxlen=CONFIG['history_len'])
# phi history: store stats of phi before each HMC update
phi_history = deque(maxlen=CONFIG['history_len'])
# optionally store H history (hamiltonian)
H_history = deque(maxlen=CONFIG['history_len'])

# ---------------------------
# Action / Force / Energy
# ---------------------------
def calculate_action(phi):
    """
    Compute lattice action S[phi].
    We use:
      S = sum_x [ 1/2 * sum_mu (phi_x - phi_{x+mu})^2 + 1/2 * m2 * phi_x^2 + lam * phi_x^4 ]
    phi shape: (batch, 1, L, L)
    returns: action per chain shape (batch,)
    """
    # neighbor shifts
    phi_right = roll(phi, 1, axis=3)
    phi_left  = roll(phi, -1, axis=3)
    phi_up    = roll(phi, -1, axis=2)
    phi_down  = roll(phi, 1, axis=2)

    # kinetic: 1/2 * sum_mu (phi_x - phi_{x+mu})^2
    kin_right = 0.5 * (phi - phi_right) ** 2
    kin_up    = 0.5 * (phi - phi_up) ** 2
    kinetic = kin_right + kin_up  # sum over the two directions
    potential = 0.5 * CONFIG['m2'] * (phi ** 2) + CONFIG['lam'] * (phi ** 4)

    action_density = kinetic + potential  # shape (batch,1,L,L)
    action = np.sum(action_density, axis=(1,2,3))  # per chain
    return action

def calculate_kinetic_energy(p):
    """Kinetic energy of momentum p: 1/2 sum p^2 per chain."""
    return 0.5 * np.sum(p ** 2, axis=(1,2,3))

def calculate_hamiltonian(phi, p):
    """Total Hamiltonian H = KineticEnergy(p) + Action(phi)"""
    return calculate_kinetic_energy(p) + calculate_action(phi)

def get_force(phi, do_record=True):
    """
    Compute force = - dS/dphi, split into three contributions:
      Kinetic part (from neighbor couplings),
      Mass part,
      Lambda part.
    We use analytic derivative of the action above:
      dS/dphi = sum_mu (2 phi_x - phi_{x+mu} - phi_{x-mu}) + m2 * phi + 4 * lam * phi^3
    so force = -( ... )
    Return force (shape same as phi) and optionally record splitted stats to force_history.
    """
    phi_right = roll(phi, 1, axis=3)
    phi_left  = roll(phi, -1, axis=3)
    phi_up    = roll(phi, -1, axis=2)
    phi_down  = roll(phi, 1, axis=2)

    # sum of neighbors
    neighbors_sum = phi_right + phi_left + phi_up + phi_down

    # derivative parts
    # kinetic derivative: sum_mu (2 phi - phi_{x+mu} - phi_{x-mu}) -> here mu=2 -> 4 phi - neighbors_sum
    dK = 4.0 * phi - neighbors_sum  # shape (batch,1,L,L)

    # mass derivative: m2 * phi
    dM = CONFIG['m2'] * phi

    # lambda derivative: 4 * lam * phi^3
    dL = 4.0 * CONFIG['lam'] * (phi ** 3)

    # full gradient
    grad = dK + dM + dL

    # force is negative gradient
    force = -grad

    # record history (stats) if requested
    if do_record:
        K_stats = stats_of_array(dK)
        M_stats = stats_of_array(dM)
        L_stats = stats_of_array(dL)
        force_history.append({
            'K': K_stats,
            'M': M_stats,
            'L': L_stats
        })

        # diagnostics: check abnormal
        threshold = CONFIG['diagnostic_threshold']
        def is_bad(s):
            return s['nan'] or s['inf'] or s['max_abs'] > threshold

        if is_bad(K_stats) or is_bad(M_stats) or is_bad(L_stats):
            print("\n============================")
            print("⚠️  Force diagnostic triggered (NaN/Inf/TooLarge). Printing last history:")
            print("============================\n")
            # print full force history
            for idx, snap in enumerate(list(force_history)):
                print(f"[history -{len(force_history)-idx}]")
                for key in ('K','M','L'):
                    s = snap[key]
                    print(f"  {key}: max_abs={s['max_abs']:.4e}, min={s['min']:.4e}, max={s['max']:.4e}, mean={s['mean']:.4e}, nan={s['nan']}, inf={s['inf']}")
                print()
            print("Explosion source(s):")
            if is_bad(K_stats): print("  Kinetic (dK) abnormal")
            if is_bad(M_stats): print("  Mass (dM) abnormal")
            if is_bad(L_stats): print("  Lambda (dL) abnormal")
            # also print phi history for context
            print("\nLast phi history (stats):")
            for idx, s in enumerate(list(phi_history)):
                print(f"  [phi hist -{len(phi_history)-idx}] max_abs={s['max_abs']:.4e}, min={s['min']:.4e}, mean={s['mean']:.4e}")
            print()
            raise RuntimeError("Force exploded; see printed history.")

    return force

# ---------------------------
# Leapfrog integrator
# ---------------------------
def get_velocity(p):
    """Velocity = p (since mass=1)."""
    return p

def leap_frog(phi, p, tao, record_force=True):
    """
    Standard leapfrog integrator:
      p <- p + (eps/2) * force(phi)
      loop:
        phi <- phi + eps * velocity(p)
        p <- p + eps * force(phi)  (except last iteration)
      p <- p + (eps/2) * force(phi)
    returns new (phi_new, p_new)
    """
    eps = tao / CONFIG['leap_frog_step']
    # half-step momentum update
    p_new = p + (eps / 2.0) * get_force(phi, do_record=record_force)
    phi_new = phi.copy()

    for i in range(CONFIG['leap_frog_step']):
        phi_new = phi_new + eps * get_velocity(p_new)
        if i == CONFIG['leap_frog_step'] - 1:
            break
        p_new = p_new + eps * get_force(phi_new, do_record=record_force)

    p_new = p_new + (eps / 2.0) * get_force(phi_new, do_record=record_force)
    return phi_new, p_new

# ---------------------------
# Time-reversal (momentum reversal) test
# ---------------------------
def time_reversal_test(phi, p, tao):
    """
    Perform: forward leapfrog -> flip momentum sign -> backward leapfrog (same tao & steps)
    Check how close phi_back is to original phi and Hamiltonian difference.
    Returns a dict of diagnostics.
    """
    # forward
    phi_fwd, p_fwd = leap_frog(phi, p, tao, record_force=False)  # do not pollute history
    # reverse momentum
    p_rev = -p_fwd
    # backward integration (same tao)
    phi_back, p_back = leap_frog(phi_fwd, p_rev, tao, record_force=False)

    # compare
    phi_diff = phi_back - phi
    max_abs_phi_diff = float(np.max(np.abs(phi_diff)))
    mean_abs_phi_diff = float(np.mean(np.abs(phi_diff)))

    H_initial = calculate_hamiltonian(phi, p)
    H_fwd = calculate_hamiltonian(phi_fwd, p_fwd)
    H_back = calculate_hamiltonian(phi_back, p_back)

    return {
        'max_abs_phi_diff': max_abs_phi_diff,
        'mean_abs_phi_diff': mean_abs_phi_diff,
        'H_initial_mean': float(np.mean(H_initial)),
        'H_fwd_mean': float(np.mean(H_fwd)),
        'H_back_mean': float(np.mean(H_back)),
        'H_diff_fwd_init': float(np.mean(H_fwd - H_initial)),
        'H_diff_back_init': float(np.mean(H_back - H_initial)),
    }

# ---------------------------
# HMC step (with diagnostics, phi history, H history)
# ---------------------------
def HMC_step(phi, p, tao, step_index):
    """
    Do an HMC proposal with diagnostics:
      - compute old H
      - do leapfrog
      - check phi_new for nan
      - compute new H, Metropolis accept/reject
      - record phi stats to phi_history
      - run time-reversal test occasionally and print diagnostics
    returns phi_final (accepted/kept)
    """
    H_old = calculate_hamiltonian(phi, p)
    # record H_old stats
    H_history.append(stats_of_array(H_old))

    # record phi stats before update
    phi_stats = stats_of_array(phi)
    phi_history.append(phi_stats)

    # do leapfrog (force recording turned on so history accumulates)
    phi_new, p_new = leap_frog(phi, p, tao, record_force=True)

    # immediate NaN check on phi_new
    if np.isnan(phi_new).any():
        print("NaN found in phi_new at step", step_index)
        print("NaN positions:", np.argwhere(np.isnan(phi_new)))
        # print last histories
        print("\n=== force history ===")
        for idx, snap in enumerate(list(force_history)):
            print(f"[force hist -{len(force_history)-idx}] K max_abs={snap['K']['max_abs']:.4e}, M max_abs={snap['M']['max_abs']:.4e}, L max_abs={snap['L']['max_abs']:.4e}")
        print("\n=== phi history ===")
        for idx, s in enumerate(list(phi_history)):
            print(f"[phi hist -{len(phi_history)-idx}] max_abs={s['max_abs']:.4e}, mean={s['mean']:.4e}")
        raise RuntimeError("NaN in phi_new detected.")

    H_new = calculate_hamiltonian(phi_new, p_new)
    delta_H = H_new - H_old  # shape (batch,)

    # Metropolis accept per chain
    prob = np.exp(-delta_H)
    # in case of overflow to inf/NaN in prob, clip
    prob = np.where(np.isfinite(prob), prob, 0.0)
    rand_num = np.random.rand(CONFIG['batch_size'])
    accept_mask = (rand_num < prob)

    if step_index % 100 == 0:
        print(f"Step {step_index}, accept ratio: {accept_mask.mean()*100:.2f}%")

    # Momentum reversal/time-reversal test occasionally (for debugging)
    if step_index % 200 == 0:
        tr = time_reversal_test(phi, p, tao)
        print(f"[TimeReversal Test @ step {step_index}] max|phi_back-phi|={tr['max_abs_phi_diff']:.4e}, mean|...|={tr['mean_abs_phi_diff']:.4e}")
        print(f" H mean initial={tr['H_initial_mean']:.6e}, fwd={tr['H_fwd_mean']:.6e}, back={tr['H_back_mean']:.6e}")
        print(f" H_diffs: fwd-init={tr['H_diff_fwd_init']:.4e}, back-init={tr['H_diff_back_init']:.4e}")

    # apply accept/reject
    mask = accept_mask.reshape((-1,1,1,1))
    phi_final = np.where(mask, phi_new, phi)

    return phi_final

# ---------------------------
# Observables: correlation as before
# ---------------------------
def get_2_point_correlation(ensemble, t_shift, space_shift):
    """
    ensemble shape: (n_saves, batch, 1, L, L)
    returns averaged 2-pt connected correlator for given shifts
    """
    ensemble_shifted = roll(roll(ensemble, shift=t_shift, axis=3), shift=space_shift, axis=4)
    corr = ensemble_shifted * ensemble
    corr_grid_mean = corr.mean(axis=(2,3,4))
    corr_ensemble_mean = corr_grid_mean.mean(axis=1).mean(axis=0)
    return corr_ensemble_mean

def get_expected_phi(ensemble):
    expected_phi = ensemble.mean(axis=(2,3,4)).mean(axis=1).mean(axis=0)
    return expected_phi

def get_connected_2_point_correlation(ensemble, t_shift, space_shift):
    corr_ensemble_mean = get_2_point_correlation(ensemble, t_shift, space_shift)
    expected_phi_ = get_expected_phi(ensemble)
    expected_phi_shifted = get_expected_phi(roll(roll(ensemble, shift=t_shift, axis=3), shift=space_shift, axis=4))
    return corr_ensemble_mean - expected_phi_ * expected_phi_shifted

def calculate_G_t_list(ensemble):
    G_t_list = []
    for t in range(CONFIG['L']):
        G_t_l_list = []
        for l in range(CONFIG['L']):
            conn = get_connected_2_point_correlation(ensemble, t, l)
            G_t_l_list.append(conn)
        G_t_mean = np.stack(G_t_l_list).mean()
        G_t_list.append(G_t_mean)
    return np.array(G_t_list)

# ---------------------------
# Main sampling loop
# ---------------------------
def main():
    # initialize random phi and seed for reproducibility
    np.random.seed(42)
    phi = np.random.randn(CONFIG['batch_size'], 1, CONFIG['L'], CONFIG['L'])
    ensemble = []

    # thermalization
    print("Starting thermalization...")
    for i in range(CONFIG['thermal_steps']):
        p = np.random.randn(CONFIG['batch_size'], 1, CONFIG['L'], CONFIG['L'])
        phi_new = HMC_step(phi, p, CONFIG['tao'], i)
        phi = phi_new

    # sampling
    print("Starting sampling...")
    for i in range(CONFIG['n_samples']):
        p = np.random.randn(CONFIG['batch_size'], 1, CONFIG['L'], CONFIG['L'])
        phi_new = HMC_step(phi, p, CONFIG['tao'], i + CONFIG['thermal_steps'])
        phi = phi_new
        if i % CONFIG['save_steps'] == 0:
            ensemble.append(phi.copy())

    ensemble_tensor = np.stack(ensemble, axis=0)  # shape (n_saves, batch, 1, L, L)
    print("Sampling finished. Ensemble shape:", ensemble_tensor.shape)

    # compute G_t
    G_t = calculate_G_t_list(ensemble_tensor)
    plt.plot(G_t, marker='o')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('G_t')
    plt.title('2-point function of 2-dim lattice phi^4 theory')
    plt.show()

if __name__ == '__main__':
    main()
