import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_tfim_sweep_circuit,
    get_tfim_ham,
    get_circuit_energy,
    get_min_gap,
    cooling_circ,
    save_sweep,
)

from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

dry_run = True
add_cooling = True


for n_qubits in range(9, 11):
    for j in np.linspace(0.9, 1, 2):
        ham_j, ham_h = get_tfim_ham(n_qubits, 1, j, periodic=False, split_parts=True)
        hamiltonian = ham_j + ham_h

        min_gap = get_min_gap(hamiltonian)

        total_time = 1 / (min_gap**3)
        n_steps = 5 * hamiltonian.num_qubits
        mult = 10

        circuit = get_tfim_sweep_circuit(
            ham_h, ham_j + ham_h, total_time=mult * total_time, n_steps=mult * n_steps
        )

        if add_cooling:
            final_circuit = QuantumCircuit(hamiltonian.num_qubits + 1)
            final_circuit.compose(circuit, range(hamiltonian.num_qubits), inplace=True)
            final_circuit.compose(
                cooling_circ(
                    ham=hamiltonian, omega=0.3303173818894809, weaken_coupling=10
                ),
                range(hamiltonian.num_qubits + 1),
                inplace=True,
            )

        else:
            final_circuit = circuit
        ground_energy = get_ground_energy(hamiltonian)

        final_energy = get_circuit_energy(hamiltonian, final_circuit)
        print(
            f"final energy error is {np.abs(ground_energy-final_energy)/np.abs(ground_energy)}"
        )
        if not dry_run:
            save_sweep(
                f"from_xx/sweep/{uuid4()}",
                hamiltonian,
                final_circuit,
                ground_energy,
            )
