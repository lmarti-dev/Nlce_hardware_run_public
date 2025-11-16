import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
    estimate_observable_1qp_initial_state,
    get_tfim_sweep_circuit,
    circuit_to_matrix,
    apply_parameters,
    get_min_gap,
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from typing import Callable

# Ground state energy of TFIM using VQE with HVA


def get_cost_offdiag(
    circuit: QuantumCircuit, hamiltonian: SparsePauliOp
) -> Callable[[np.ndarray], float]:
    def cost(xk: np.ndarray) -> float:
        a_mat = estimate_observable_1qp_initial_state(circuit, xk, hamiltonian)
        b_mat = estimate_observable_1qp_initial_state(circuit, xk, hamiltonian**2)
        return np.abs(np.trace(b_mat) - np.sum(np.abs(a_mat) ** 2))

    return cost


def get_cost_sweep(
    vqe_circ: QuantumCircuit,
    ham_start: SparsePauliOp,
    ham_end: SparsePauliOp,
) -> Callable[[np.ndarray], float]:
    min_gap = get_min_gap(hamiltonian)
    total_time = 0.7 / (min_gap**2)
    n_steps = 4 * hamiltonian.num_qubits

    sweep_circuit = get_tfim_sweep_circuit(
        ham_start,
        ham_end,
        total_time=total_time,
        n_steps=int(n_steps),
    )
    sweep_mat = circuit_to_matrix(sweep_circuit)
    identity = np.eye(2**circuit.num_qubits)

    def cost(xk: np.ndarray) -> float:

        vqe_mat = circuit_to_matrix(apply_parameters(vqe_circ, xk))

        return np.sum(np.abs(sweep_mat.conj().T @ vqe_mat - identity))

    return cost


if __name__ == "__main__":

    dry_run = True

    for n_qubits in (4, 5):
        for J in (0.8, 0.9, 1):
            print(f"{n_qubits} qubits, J/h={J}")
            periodic = False
            h = 1
            ham_j, ham_h = TFIM(n_qubits, h, J, periodic=False, split_parts=True)
            hamiltonian = ham_j + ham_h
            n_layers = int(np.ceil(hamiltonian.num_qubits / 2))

            circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)

            # circuit = get_custom_vqe(4, n_layers=2)
            ground_energy = get_ground_energy(hamiltonian)

            cost = get_cost_sweep(circuit, ham_j, hamiltonian)

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
