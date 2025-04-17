from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
import numpy as np
from .statevector import estimate_observables


def iterate_over_1_qp(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    xk: np.ndarray,
    pow: int = 1,
) -> float:
    n_qubits = hamiltonian.num_qubits
    energy = 0
    for q in range(n_qubits):
        out = QuantumCircuit(n_qubits)
        out.x(q)
        result = estimate_observables(
            circuit=out.compose(circuit), observables=[hamiltonian], param_sweep=xk
        )
        energy += result.data.evs[0] ** pow
    return energy


def get_minimize_1qp_subspace_energy(
    hamiltonian: SparsePauliOp, circuit: QuantumCircuit
) -> callable:

    def cost(xk: np.ndarray) -> float:
        return iterate_over_1_qp(hamiltonian=hamiltonian, circuit=circuit, xk=xk)

    return cost


def get_minimize_variance(
    hamiltonian: SparsePauliOp, circuit: QuantumCircuit
) -> callable:
    ham_squared = hamiltonian**2

    def cost(xk: np.ndarray) -> float:
        return iterate_over_1_qp(
            hamiltonian=ham_squared, circuit=circuit, xk=xk
        ) - iterate_over_1_qp(hamiltonian=ham_squared, circuit=circuit, xk=xk, pow=2)

    return cost
