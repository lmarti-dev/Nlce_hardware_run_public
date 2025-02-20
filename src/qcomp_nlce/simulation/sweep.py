import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_tfim_sweep_circuit,
    get_tfim_ham,
    get_circuit_energy,
    get_min_gap,
    save_sweep,
)

for n_qubits in range(9, 11):
    for j in np.linspace(0.5, 1, 5):
        ham_j, ham_h = get_tfim_ham(n_qubits, 1, j, periodic=False, split_parts=True)
        hamiltonian = ham_j + ham_h

        min_gap = get_min_gap(hamiltonian)

        total_time = 1 / (min_gap**3)
        n_steps = 5 * hamiltonian.num_qubits
        mult = 10

        circuit = get_tfim_sweep_circuit(
            ham_j, ham_h, total_time=mult * total_time, n_steps=mult * n_steps
        )
        ground_energy = get_ground_energy(hamiltonian)

        print(
            f"The circuit depth is {circuit.depth()} for {hamiltonian.num_qubits} qubits"
        )

        final_energy = get_circuit_energy(hamiltonian, circuit)
        print(
            f"final energy error is {np.abs(ground_energy-final_energy)/np.abs(ground_energy)}"
        )

        save_sweep(
            f"scan_jh/sweep/{uuid4()}",
            hamiltonian,
            circuit,
            ground_energy,
        )
