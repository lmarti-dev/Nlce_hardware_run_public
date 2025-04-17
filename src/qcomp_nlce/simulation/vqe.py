import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
)

dry_run = True

for n_qubits in (9, 10):
    for J in (0.8, 0.9, 1):
        periodic = False
        h = 1
        hamiltonian = TFIM(n_qubits, h, J, periodic=periodic)
        n_layers = int(np.ceil(hamiltonian.num_qubits / 2))

        circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)
        # circuit = get_custom_vqe(4, n_layers=2)
        ground_energy = get_ground_energy(hamiltonian)

        print(f"The circuit depth is {circuit.depth()}")
        print(f"The number of parameters is {len(circuit.parameters)}")
        energy_values, params = optimize_circuit(circuit, hamiltonian)
        print(
            f"final energy error is {np.abs(ground_energy-energy_values[-1])/np.abs(ground_energy)}"
        )

        save_optimized_vqe(
            f"early_comp/vqe/{uuid4()}",
            hamiltonian,
            circuit,
            params,
            ground_energy,
            dry_run,
        )
