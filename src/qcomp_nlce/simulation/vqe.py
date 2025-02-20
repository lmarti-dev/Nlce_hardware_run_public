import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    get_tfim_ham,
    optimize_circuit,
    save_optimized_vqe,
)


for n_qubits in (10,):
    periodic = False
    hamiltonian = get_tfim_ham(n_qubits, 1, 0.7, periodic=periodic)
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
        f"new_test/vqe/{uuid4()}",
        hamiltonian,
        circuit,
        params,
        ground_energy,
    )
