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
)


dry_run = False

for n_qubits in (9, 10):
    for J in (0.8, 0.9, 1):
        ham_j, ham_h = get_tfim_ham(n_qubits, 1, J, periodic=False, split_parts=True)
        hamiltonian = ham_j + ham_h

        min_gap = get_min_gap(hamiltonian)

        total_time = 1 / (min_gap**3)
        n_steps = 5 * hamiltonian.num_qubits
        mult = 10

        circuit = get_tfim_sweep_circuit(
            ham_h, ham_j + ham_h, total_time=mult * total_time, n_steps=mult * n_steps
        )
        ground_energy = get_ground_energy(hamiltonian)

        print(
            f"The circuit depth is {circuit.depth()} for {hamiltonian.num_qubits} qubits j/h={J:.3f}"
        )

        final_energy = get_circuit_energy(hamiltonian, circuit)
        print(
            f"final energy error is {np.abs(ground_energy-final_energy)/np.abs(ground_energy)}"
        )
        save_sweep(
            f"9q_10q_gs/sweep/{uuid4()}", hamiltonian, circuit, ground_energy, dry_run
        )
