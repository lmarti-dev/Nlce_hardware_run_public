import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_tfim_sweep_circuit,
    get_tfim_ham,
    get_circuit_energy,
    get_min_gap,
    cooling_step,
    save_sweep,
    subspace_diagonalize,
    statevector_sample_unique,
)

from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

dry_run = True

qubit_sizes = list(range(3, 7))
j_vals = np.linspace(0.9, 1, 2)
coolings = (False, True)

errors = np.zeros((len(qubit_sizes), len(j_vals), 2, 2))

for indq, n_qubits in enumerate(qubit_sizes):
    for indj, j in enumerate(j_vals):
        for indc, add_cooling in enumerate(coolings):
            ham_j, ham_h = get_tfim_ham(
                n_qubits, 1, j, periodic=False, split_parts=True
            )
            hamiltonian = ham_j + ham_h

            min_gap = get_min_gap(hamiltonian)

            total_time = 1 / (min_gap**3)
            n_steps = 5 * hamiltonian.num_qubits
            n_steps = 5
            mult = 1

            circuit = get_tfim_sweep_circuit(
                ham_h,
                ham_j + ham_h,
                total_time=mult * total_time,
                n_steps=int(mult * n_steps),
            )

            if add_cooling:
                final_circuit = QuantumCircuit(hamiltonian.num_qubits + 1)
                final_circuit.compose(
                    circuit, range(hamiltonian.num_qubits), inplace=True
                )
                final_circuit.compose(
                    cooling_step(
                        ham=hamiltonian,
                        omega=0.3303173818894809,
                        weaken_coupling=10,
                        coupler_type="proj",
                    ),
                    range(hamiltonian.num_qubits + 1),
                    inplace=True,
                )

            else:
                final_circuit = circuit
            ground_energy = get_ground_energy(hamiltonian)

            def relerr(energy):
                return np.abs(ground_energy - energy) / np.abs(ground_energy)

            print(
                f"Cooling: {add_cooling} Depth {final_circuit.depth()} for {hamiltonian.num_qubits} qubits j/h={j:.3f}"
            )
            final_energy = get_circuit_energy(hamiltonian, final_circuit)
            unique_bitstrings = statevector_sample_unique(final_circuit, 1000)
            if add_cooling:
                unique_bitstrings = [
                    u[-1:0:-1] for u in unique_bitstrings if u[0] == "0"
                ]
            subspace_energy = subspace_diagonalize(
                unique_bitstrings=unique_bitstrings, ham=hamiltonian
            )
            err0 = relerr(final_energy)
            err1 = relerr(subspace_energy)
            print(f"sv error {err0} sub error {err1}")
            errors[indq, indj, indc, 0] = err0
            errors[indq, indj, indc, 1] = err1
            if not dry_run:
                save_sweep(
                    f"from_xx/sweep/{uuid4()}",
                    hamiltonian,
                    final_circuit,
                    ground_energy,
                )

mark = "xods"
col = "rbg"
fig, ax = plt.subplots()
for j_val in range(errors.shape[1]):
    for cool in range(errors.shape[2]):
        ax.plot(
            qubit_sizes,
            errors[:, j_val, cool, 1],
            marker=mark[j_val],
            color=col[cool],
            markersize=12,
            label=f"{j_vals[j_val]}j/h, cool: {bool(coolings[cool])}",
        )

ax.grid(True)
ax.legend()
ax.set_ylabel("relative error")
ax.set_xlabel("qubits")
ax.set_yscale("log")

plt.show()
