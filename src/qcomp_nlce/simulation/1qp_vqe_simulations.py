import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
    estimate_observable_1qp_initial_state,
)


if __name__ == "__main__":

    dry_run = False

    for n_qubits in (4, 5):
        for J in (0.8, 0.9, 1):
            print(f"{n_qubits} qubits, J/h={J}")
            periodic = False
            h = 1
            hamiltonian = TFIM(n_qubits, h, J, periodic=periodic)
            n_layers = int(np.ceil(hamiltonian.num_qubits / 2))

            circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)

            def cost(xk: np.ndarray) -> float:
                a_mat = estimate_observable_1qp_initial_state(circuit, xk, hamiltonian)
                b_mat = estimate_observable_1qp_initial_state(
                    circuit, xk, hamiltonian**2
                )
                return np.abs(np.trace(b_mat) - np.sum(np.abs(a_mat) ** 2))

            # circuit = get_custom_vqe(4, n_layers=2)
            ground_energy = get_ground_energy(hamiltonian)

            print(f"The circuit depth is {circuit.depth()}")
            print(f"The number of parameters is {len(circuit.parameters)}")
            cost_values, params = optimize_circuit(circuit, hamiltonian, cost=cost)
            print(f"final cost is {np.abs(cost_values[-1])}")

            save_optimized_vqe(
                f"1qp_small/vqe/{uuid4()}",
                hamiltonian,
                circuit,
                params,
                ground_energy,
                dry_run,
            )
