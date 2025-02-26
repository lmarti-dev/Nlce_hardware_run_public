from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit, AncillaRegister
import numpy as np


def cooling_circ(
    ham: SparsePauliOp, omega: float, weaken_coupling: float
) -> QuantumCircuit:
    n_qubits = ham.num_qubits

    alpha = omega / (weaken_coupling * n_qubits)
    anc_ham = 0.5 * (
        SparsePauliOp("I" * (n_qubits + 1)) - SparsePauliOp("I" * n_qubits + "Z")
    )
    circuit = QuantumCircuit(ham.num_qubits + 1)
    coupler = SparsePauliOp("I" * (n_qubits - 1) + "XX") + SparsePauliOp(
        "I" * (n_qubits - 1) + "YY"
    )
    evolution_time = np.pi / alpha
    circuit.append(
        PauliEvolutionGate(
            SparsePauliOp("I").expand(ham) + coupler * alpha + omega * anc_ham,
            evolution_time,
        ),
        range(ham.num_qubits + 1),
    )
    return circuit
