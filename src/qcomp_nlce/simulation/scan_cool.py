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
    ground_state_fidelity,
)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, SparsePauliOp
import matplotlib.pyplot as plt

dry_run = True

n_qubits = 5
n_steps = 10
errors = np.zeros((n_steps, 2))
j = 1

ham_j, ham_h = get_tfim_ham(n_qubits, 1, j, periodic=False, split_parts=True)
hamiltonian = ham_j + ham_h

min_gap = get_min_gap(hamiltonian)

n_trotter_cooling = 40

mat = hamiltonian.to_matrix(sparse=False)
energies = np.linalg.eigvalsh(mat)

qreg = QuantumRegister(hamiltonian.num_qubits + 1)
circuit = QuantumCircuit(qreg)
ground_energy = get_ground_energy(hamiltonian)


for step in range(n_steps):
    k = n_steps - step - 1
    circuit.compose(
        cooling_step(
            ham=hamiltonian,
            omega=energies[k] - energies[0],
            weaken_coupling=5,
            n_steps=n_trotter_cooling,
            coupler_type="XX",
        ),
        range(hamiltonian.num_qubits + 1),
        inplace=True,
    )
    circuit.reset(qreg[-1])

    def relerr(energy):
        return np.abs(ground_energy - energy) / np.abs(ground_energy)

    final_energy = get_circuit_energy(hamiltonian, circuit, use_estimator=True)
    unique_bitstrings = statevector_sample_unique(circuit, 1000, use_sampler=True)
    tmp_unique_bitstrings = [u[1:] for u in unique_bitstrings if u[0] == "0"]
    if tmp_unique_bitstrings != []:
        unique_bitstrings = tmp_unique_bitstrings
    subspace_energy = subspace_diagonalize(
        unique_bitstrings=unique_bitstrings, ham=hamiltonian
    )
    final_sys_sv = partial_trace(Statevector(circuit), [circuit.num_qubits - 1])
    print(f"final fidelity: {ground_state_fidelity(final_sys_sv,hamiltonian):.4e}")
    err0 = relerr(final_energy)
    err1 = relerr(subspace_energy)
    print(f"sv e {err0:4e} sub e {err1:4e}")
    errors[step, 0] = err0
    errors[step, 1] = err1
if not dry_run:
    save_sweep(
        f"scan_cool/sweep/{uuid4()}",
        hamiltonian,
        circuit,
        ground_energy,
    )


print(f"Depth {circuit.depth()} for {hamiltonian.num_qubits} qubits j/h={j:.3f}")

mark = "xods+"
col = "rbg"
fig, ax = plt.subplots()
ax.plot(
    range(n_steps),
    errors[:, 0],
    markersize=12,
    label="sv err",
)
ax.plot(
    range(n_steps),
    errors[:, 1],
    markersize=12,
    label="sub err",
)

ax.grid(True)
ax.legend()
ax.set_ylabel("relative error")
ax.set_xlabel("qubits")
ax.set_yscale("log")

plt.show()
