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


def save_optimized_vqe(
    dirname: str,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    weights: np.ndarray,
    ground_energy: float,
):
    home = Path(__file__).parent.parent
    save_dir = Path(home, f"res_sim/{dirname}/")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(Path(save_dir, "circuit.qpy"), "wb+") as fd:
        qpy.dump(circuit, fd)
    with open(Path(save_dir, "weights.npy"), "wb+") as fd:
        np.save(fd, weights)
    with open(Path(save_dir, "ground_energy.npy"), "wb+") as fd:
        np.save(fd, np.array(ground_energy))
    with open(Path(save_dir, "hamiltonian.npy"), "wb+") as fd:
        np.save(fd, hamiltonian.to_list(True))


def load_optimized_vqe(dirname: str):
    home = Path(__file__).parent.parent
    save_dir = Path(home, f"res_sim/{dirname}/")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(Path(save_dir, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)[0]
    with open(Path(save_dir, "weights.npy"), "rb") as fd:
        weights = np.load(fd)
    with open(Path(save_dir, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(save_dir, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, weights, ground_energy


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
