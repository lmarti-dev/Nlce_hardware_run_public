import numpy as np
from uuid import uuid4

from src.qcomp_nlce.utils import (
    get_ground_energy,
    get_tfim_sweep_circuit,
    TFIM,
    get_circuit_energy,
    get_min_gap,
    cooling_step,
    save_sweep,
    subspace_diagonalize,
    statevector_sample_unique,
    ground_state_fidelity,
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

dry_run = True

qubit_sizes = (9, 10)
j_vals = (0.8, 0.9, 1)
coolings = (False, True)


errors = np.zeros((len(qubit_sizes), len(j_vals), 2, 2))

for indq, n_qubits in enumerate(qubit_sizes):
    for indj, j in enumerate(j_vals):
        for indc, add_cooling in enumerate(coolings):
            ham_j, ham_h = TFIM(n_qubits, 1, j, periodic=False, split_parts=True)
            hamiltonian = ham_j + ham_h

            min_gap = get_min_gap(hamiltonian)

            total_time = 1 / (min_gap**3)
            n_sweep_steps = 5 * hamiltonian.num_qubits
            n_sweep_steps = 5
            n_cooling_steps = 1
            mult = 1

            circuit = get_tfim_sweep_circuit(
                ham_h,
                ham_j + ham_h,
                total_time=mult * total_time,
                n_steps=int(mult * n_sweep_steps),
            )

            if add_cooling:
                mat = hamiltonian.to_matrix(sparse=True)
                energies, _ = eigsh(mat, k=2, which="SA")
                final_circuit = QuantumCircuit(hamiltonian.num_qubits + 1)
                final_circuit.compose(
                    circuit, range(hamiltonian.num_qubits), inplace=True
                )
                final_circuit.compose(
                    cooling_step(
                        ham=hamiltonian,
                        omega=energies[1] - energies[0],
                        weaken_coupling=5,
                        n_steps=n_cooling_steps,
                        coupler_type="XX+YY",
                    ),
                    range(hamiltonian.num_qubits + 1),
                    inplace=True,
                )

            else:
                final_circuit = circuit
            ground_energy = get_ground_energy(hamiltonian)

            def relerr(energy):
                return np.abs(ground_energy - energy) / np.abs(ground_energy)

            msg = "Cooling" if add_cooling else "Sweeping"
            print(
                f"{msg} Depth {final_circuit.depth()} for {hamiltonian.num_qubits} qubits j/h={j:.3f}"
            )
            final_energy = get_circuit_energy(hamiltonian, final_circuit)
            unique_bitstrings = statevector_sample_unique(final_circuit, 1000)
            if add_cooling:
                tmp_unique_bitstrings = [
                    u[1:] for u in unique_bitstrings if u[0] == "1"
                ]
                if tmp_unique_bitstrings != []:
                    unique_bitstrings = tmp_unique_bitstrings
            subspace_energy = subspace_diagonalize(
                unique_bitstrings=unique_bitstrings, ham=hamiltonian
            )
            if add_cooling:
                final_sv = partial_trace(
                    Statevector(final_circuit), [final_circuit.num_qubits - 1]
                )
            else:
                final_sv = Statevector(final_circuit)
            print(f"final fidelity: {ground_state_fidelity(final_sv,hamiltonian):.4e}")
            err0 = relerr(final_energy)
            err1 = relerr(subspace_energy)
            print(f"Statevector {err0:4e} subspace {err1:4e}")
            errors[indq, indj, indc, 0] = err0
            errors[indq, indj, indc, 1] = err1
            save_sweep(
                f"from_xx/sweep/{uuid4()}",
                hamiltonian,
                final_circuit,
                ground_energy,
                dry_run,
            )

mark = "xods+"
col = "rbg"
fig, ax = plt.subplots()
for cool in range(errors.shape[2]):
    for j_val in range(errors.shape[1]):
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
