from qcomp_nlce.utils import (
    TFIM,
    get_min_gap,
    get_tfim_sweep_circuit,
    get_ground_energy,
    get_max_energy,
)
from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    measure_ground_state_energy_subspace_sampling_v2,
    measure_ground_state_energy_estimator,
)
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, c, d):
    return a * np.exp(-c * x) + d


def get_prop_of_counts(counts, p):
    all_counts = out["statevector"]["counts"]
    keys = sorted(list(all_counts.keys()), key=lambda x: -all_counts[x])
    counts = {k: all_counts[k] for k in keys[: int(p * len(keys))]}
    return counts


n_qubits = 10
J = 1
n_shots = 10000

ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
hamiltonian = ham_j + ham_h
min_gap = get_min_gap(hamiltonian)
ground_energy = get_ground_energy(hamiltonian)
max_ham_end = get_max_energy(hamiltonian)
max_ham_start = get_max_energy(ham_h)
# the most important thing is to have enough steps!!!
# the total time can be low-ish


n_steps_list = np.arange(2, 100, 5)

energies = np.zeros_like(n_steps_list)
for jjj, n_steps in enumerate(n_steps_list):

    alpha = abs(max_ham_start) + abs(max_ham_end)
    total_time = n_steps / alpha
    print(f"{jjj:0>3} -- total time: {total_time:.5e}, n_steps: {n_steps:5.0f}")
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )

    # out = measure_ground_state_energy_subspace_sampling_v2(
    #     hamiltonian=hamiltonian,
    #     ground_energy=ground_energy,
    #     circuit=circuit,
    #     which_backend="statevector",
    #     n_shots=n_shots,
    #     provider_name="IBM",
    # )
    out = measure_ground_state_energy_estimator(
        hamiltonian=hamiltonian,
        target_energy=ground_energy,
        circuit=circuit,
        n_shots=n_shots,
        which_backend="statevector",
    )

    energies[jjj] = out["statevector"]["energy"]


popt, pcov = curve_fit(func, n_steps_list, energies, p0=(1, 3, 1))
f_xx = np.linspace(1, 1000, 100)

f_yy = func(f_xx, *popt)

fig, ax = plt.subplots()

ax.plot(n_steps_list, energies, label="SQD estimate")
ax.plot(f_xx, f_yy, label="Exponential fit")
ax.hlines(ground_energy, f_xx[0], f_xx[-1], "r", label="True ground energy")
ax.set_xlabel(f"Sweep time ({n_qubits} qubits, {n_shots} shots)")
ax.set_xscale("log")
ax.legend()
ax.set_ylabel(r"$Energy$")

print(f"Ground energy: {ground_energy:.5f}; extrapolation: {func(1e6, *popt)}")


plt.show()
