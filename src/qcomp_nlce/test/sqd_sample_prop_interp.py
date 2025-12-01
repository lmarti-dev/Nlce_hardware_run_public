from qcomp_nlce.utils import (
    TFIM,
    get_min_gap,
    get_tfim_sweep_circuit,
    get_unique_bitstrings,
    subspace_diagonalize,
    get_ground_energy,
)
from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    measure_ground_state_energy_subspace_sampling_v2,
)
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, c, d):
    return a * np.exp(-c * x) + d


n_qubits = 10
J = 1
mult = 1

ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
hamiltonian = ham_j + ham_h
min_gap = get_min_gap(hamiltonian)
ground_energy = get_ground_energy(hamiltonian)
# the most important thing is to have enough steps!!!
# the total time can be low-ish

total_time = mult / (min_gap**2)
n_steps = int(mult * n_qubits**2)
circuit = get_tfim_sweep_circuit(
    ham_h,
    ham_j + ham_h,
    total_time=total_time,
    n_steps=n_steps,
)

n_shots = 1000

out = measure_ground_state_energy_subspace_sampling_v2(
    hamiltonian=hamiltonian,
    ground_energy=ground_energy,
    circuit=circuit,
    which_backend="statevector",
    n_shots=n_shots,
    provider_name="IBM",
)

props = np.linspace(0.1, 1, 100)
energies = np.zeros_like(props)

all_counts = out["statevector"]["counts"]

keys = sorted(list(all_counts.keys()), key=lambda x: -all_counts[x])

for jjj, p in enumerate(props):
    print(f"Prop: {p}")
    counts = {k: all_counts[k] for k in keys[: int(p * len(keys))]}
    unique_bitstring = get_unique_bitstrings(counts)
    bitstring_prop = len(unique_bitstring) / (2**circuit.num_qubits)
    comp_energy = subspace_diagonalize(
        unique_bitstrings=unique_bitstring, ham=hamiltonian
    )
    energies[jjj] = comp_energy


popt, pcov = curve_fit(func, props, energies, p0=(1, 3, 1))
f_xx = np.linspace(0.1, 10, 100)

f_yy = func(f_xx, *popt)

fig, ax = plt.subplots()

ax.plot(props, energies, label="SQD estimate")
ax.plot(f_xx, f_yy, label="Exponential fit")
ax.hlines(ground_energy, f_xx[0], f_xx[-1], "r", label="True ground energy")
ax.set_xlabel(
    f"Proportion of sampled bitstrings used in SQD ({n_qubits} qubits, {n_shots} shots)"
)
ax.set_xscale("log")
ax.legend()
ax.set_ylabel(r"$Energy$")

print(f"Ground energy: {ground_energy:.5f}; extrapolation: {func(1e6, *popt)}")


plt.show()
