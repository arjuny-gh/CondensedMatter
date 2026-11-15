"""2D Ising model, Metropolis Monte Carlo (Stage 1.3). Units J = k_B = 1.

Exact critical temperature (Onsager): T_c = 2 / ln(1 + sqrt(2)) ~ 2.2692.
Checkerboard updates: spins on one sublattice have no neighbours on the same
sublattice, so they can be updated simultaneously without breaking detailed balance.
"""
import numpy as np

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))


def energy(s):
    """Total energy with periodic boundaries (each bond counted once)."""
    return -np.sum(s * (np.roll(s, 1, 0) + np.roll(s, 1, 1)))


def sweep(s, T, rng):
    L = s.shape[0]
    ii, jj = np.indices(s.shape)
    for parity in (0, 1):
        mask = (ii + jj) % 2 == parity
        nb = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
        dE = 2 * s * nb
        accept = (dE <= 0) | (rng.random(s.shape) < np.exp(-np.clip(dE, 0, None) / T))
        s[mask & accept] *= -1


def simulate(L, T, n_therm=1000, n_meas=2000, seed=0, cold=False):
    """Return per-spin mean |m|, energy e, specific heat C, susceptibility chi."""
    rng = np.random.default_rng(seed)
    s = np.ones((L, L), dtype=np.int8) if cold else rng.choice(np.array([-1, 1], dtype=np.int8), (L, L))
    for _ in range(n_therm):
        sweep(s, T, rng)
    m, e = [], []
    for _ in range(n_meas):
        sweep(s, T, rng)
        m.append(abs(s.sum()) / L**2)
        e.append(energy(s) / L**2)
    m, e = np.array(m), np.array(e)
    N = L**2
    return dict(T=T, m=m.mean(), e=e.mean(),
                C=N * e.var() / T**2, chi=N * (np.mean(m**2) - m.mean() ** 2) / T)
