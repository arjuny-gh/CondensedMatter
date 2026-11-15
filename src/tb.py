"""Tight-binding band structures (Stage 1.2).

Units: energies in eV, lengths in Angstrom. Hamiltonians are H(k) matrices
built from Bloch phases; eigenvalues give the bands E_n(k).
"""
import numpy as np


def chain_bands(k, t=1.0, a=1.0):
    """1D monatomic chain, nearest-neighbour hopping t: E = -2 t cos(k a).

    Built as a 1x1 Bloch Hamiltonian H(k) = -t (e^{ika} + e^{-ika}).
    """
    k = np.atleast_1d(k)
    return np.array([np.linalg.eigvalsh(np.array([[-t * 2 * np.cos(kk * a)]]))[0] for kk in k])


def honeycomb_H(kx, ky, t=2.7, a=1.42):
    """Graphene pi-band Hamiltonian; a = carbon-carbon distance (Angstrom).

    H = [[0, f], [f*, 0]],  f = -t * sum_j exp(i k . delta_j)
    """
    d = a * np.array([[0, 1], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]])
    f = -t * np.sum(np.exp(1j * (kx * d[:, 0] + ky * d[:, 1])))
    return np.array([[0, f], [np.conj(f), 0]])


def honeycomb_bands(kx, ky, t=2.7, a=1.42):
    return np.linalg.eigvalsh(honeycomb_H(kx, ky, t, a))


def graphene_K(a=1.42):
    """A Dirac point of graphene in this coordinate convention."""
    return np.array([4 * np.pi / (3 * np.sqrt(3) * a), 0.0])


def fermi_velocity(t=2.7, a=1.42):
    """v_F = 3 t a / (2 hbar); returns m/s. t in eV, a in Angstrom."""
    hbar = 6.582119569e-16  # eV s
    return 1.5 * t * a * 1e-10 / hbar
