import numpy as np
from pathlib import Path
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from uuid import uuid4
from qiskit.circuit.library import hamiltonian_variational_ansatz
from qiskit.quantum_info import SparsePauliOp
from qcomp_nlce.utils.hamiltonian import pretty_pauli_str
import os


def get_custom_vqe(n_qubits: int, n_layers: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for x in range(n_qubits):
            param1 = Parameter(name=str(uuid4()))
            param2 = Parameter(name=str(uuid4()))
            circuit.ry(param1, x)
            circuit.rx(param2, x)
        for x, y in zip(range(n_qubits - 1), range(1, n_qubits)):
            circuit.cx(x, y)
    return circuit


def get_hva_vqe(
    hamiltonian: SparsePauliOp, n_layers: int, unique_weights: bool = False
) -> QuantumCircuit:
    if unique_weights:
        circuit = QuantumCircuit(hamiltonian.num_qubits)
        for layer in range(n_layers):
            for term in hamiltonian:
                p = term.paulis[0]
                param = Parameter(f"l{layer}_{pretty_pauli_str(p)}")
                circuit.append(PauliEvolutionGate(term, time=param), range(len(p)))

    else:
        circuit = hamiltonian_variational_ansatz(hamiltonian, n_layers)
    return circuit


def get_givens_vqe(hamiltonian: SparsePauliOp, n_layers: int) -> QuantumCircuit:
    n_qubits = hamiltonian.num_qubits
    circuit = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits - 1):
            xx = SparsePauliOp(q * "I" + "XX" + (n_qubits - 2 - q) * "I")
            yy = SparsePauliOp(q * "I" + "YY" + (n_qubits - 2 - q) * "I")
            param = Parameter(f"l{layer}_{q}")
            circuit.append(PauliEvolutionGate(xx + yy, time=param), range(n_qubits))

    return circuit
