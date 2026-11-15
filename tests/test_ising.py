import numpy as np
from src import ising


def test_ground_state_energy():
    assert ising.energy(np.ones((8, 8), dtype=np.int8)) == -2 * 64


def test_low_T_ordered_high_T_disordered():
    assert ising.simulate(16, 1.0, 300, 300, cold=True)["m"] > 0.95
    assert ising.simulate(16, 5.0, 300, 300)["m"] < 0.25


def test_specific_heat_peak_near_onsager():
    Ts = np.linspace(1.8, 2.8, 11)
    C = [ising.simulate(16, T, 500, 1500, seed=1)["C"] for T in Ts]
    assert abs(Ts[int(np.argmax(C))] - ising.TC_EXACT) < 0.25
