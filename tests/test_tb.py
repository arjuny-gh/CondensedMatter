import numpy as np
from src import tb


def test_chain_matches_analytic():
    k = np.linspace(-np.pi, np.pi, 101)
    assert np.allclose(tb.chain_bands(k, t=1.3), -2 * 1.3 * np.cos(k), atol=1e-12)


def test_graphene_dirac_point():
    E = tb.honeycomb_bands(*tb.graphene_K())
    assert np.allclose(E, 0, atol=1e-9)


def test_graphene_linear_dispersion_and_vF():
    K = tb.graphene_K(); dk = 1e-4  # 1/Angstrom
    E = tb.honeycomb_bands(K[0] + dk, K[1])[1]       # eV
    hbar = 6.582119569e-16
    v = E / (hbar * dk * 1e10)                       # E/(hbar dk), dk converted to 1/m -> m/s
    assert abs(v - tb.fermi_velocity()) / tb.fermi_velocity() < 1e-3
    assert 0.8e6 < tb.fermi_velocity() < 1.2e6       # ~1e6 m/s in the literature
