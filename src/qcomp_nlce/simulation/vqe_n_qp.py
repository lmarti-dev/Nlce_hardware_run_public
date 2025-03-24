import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy_n_qp,
    get_givens_vqe,
    get_tfim_ham,
    optimize_circuit,
    save_optimized_vqe,
    circuit_composed_with_prep,
)


dry_run = False

for n_qubits in (9, 10):
    for J in (0.8, 0.9, 1):
        periodic = False
        h = 1
        hamiltonian = get_tfim_ham(n_qubits, h, J, periodic=periodic)
        n_layers = int(np.ceil(hamiltonian.num_qubits) / 2)

        n_qp = 1

        circuit = get_givens_vqe(hamiltonian, n_layers=n_layers)
        circuit = circuit_composed_with_prep(circuit, n_qp)
        ground_energy = get_ground_energy_n_qp(hamiltonian, n_qp)

        print(f"The number of quasi-particles is {n_qp}")
        print(f"The circuit depth is {circuit.depth()}")
        print(f"The number of parameters is {len(circuit.parameters)}")
        energy_values, params = optimize_circuit(circuit, hamiltonian)
        relerr = np.abs(ground_energy - energy_values[-1]) / np.abs(ground_energy)
        print(f"final energy error is {relerr:.5e}")

        if not dry_run:
            save_optimized_vqe(
                f"1_qp/vqe/{uuid4()}",
                hamiltonian,
                circuit,
                params,
                ground_energy,
            )
