import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    measure_matrix_elements,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
    n_qp_bitstrings,
    get_minimize_variance,
    field_pauli_sum,
)


dry_run = True

for n_qubits in (9, 10):
    for J in (0.8, 0.9, 1):
        periodic = False
        h = 1
        hamiltonian = TFIM(n_qubits, h, J, periodic=periodic)
        hamiltonian += field_pauli_sum(n_qubits, "X")

        n_layers = int(np.ceil(hamiltonian.num_qubits) / 2)

        n_qp = 1

        circuit = get_hva_vqe(hamiltonian, n_layers, True)

        print(f"The number of quasi-particles is {n_qp}")
        print(f"The circuit depth is {circuit.depth()}")
        print(f"The number of parameters is {len(circuit.parameters)}")

        cost: callable = get_minimize_variance(hamiltonian, circuit)

        energy_values, params = optimize_circuit(circuit, hamiltonian, cost)

        n_qp_bitstrings(n_qubits, n_qp, right_to_left=True)

        out = measure_matrix_elements()

        if not dry_run:
            save_optimized_vqe(
                f"1_qp/vqe/{uuid4()}",
                hamiltonian,
                circuit,
                params,
            )
