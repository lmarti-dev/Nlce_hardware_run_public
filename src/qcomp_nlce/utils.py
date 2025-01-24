from pathlib import Path
from uuid import uuid1
import os

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.circuit.library import hamiltonian_variational_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh


def get_tfim_ham(n_qubits: int, h: float, J: float, periodic: bool = False):

    hamiltonian_J = sum(
        [
            -J * SparsePauliOp(s * "I" + "XX" + (n_qubits - 2 - s) * "I")
            for s in range(n_qubits - 1)
        ]
    )
    if periodic:
        hamiltonian_J += SparsePauliOp("X" + "I" * (n_qubits - 2) + "X")
    hamiltonian_g = sum(
        [
            -h * SparsePauliOp(s * "I" + "Z" + (n_qubits - 1 - s) * "I")
            for s in range(n_qubits - 1)
        ]
    )

    hamiltonian = hamiltonian_J + hamiltonian_g

    return hamiltonian


def estimate_observables(
    circuit: QuantumCircuit, observables: list, param_sweep: np.ndarray
):
    estimator = StatevectorEstimator()
    pub = (circuit, observables, param_sweep)
    job = estimator.run([pub])
    result = job.result()[0]
    return result


def optimize_circuit(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> list:
    energy_values = []
    xks = []

    def callback(xk: np.ndarray, *args):
        energy_values.append(fun(xk=xk))
        xks.append(xk)

    def fun(xk: np.ndarray):
        result = estimate_observables(
            circuit=circuit, observables=[hamiltonian], param_sweep=xk
        )
        return result.data.evs[0]

    minimize(
        fun=fun,
        x0=np.zeros((len(circuit.parameters))),
        method="trust-constr",
        callback=callback,
        bounds=circuit.num_parameters * [(0, np.pi)],
    )
    return energy_values, xks


def get_ground_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    ground_energy, _ = eigsh(mat, k=1, which="SA")
    return ground_energy


def get_hva_vqe(hamiltonian: SparsePauliOp, n_layers: int) -> QuantumCircuit:
    return hamiltonian_variational_ansatz(hamiltonian, n_layers)


def get_custom_vqe(n_qubits: int, n_layers: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for x in range(n_qubits):
            param1 = Parameter(name=str(uuid1()))
            param2 = Parameter(name=str(uuid1()))
            circuit.ry(param1, x)
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
    home = Path(__file__).parent
    save_dir = Path(home, f"results/{dirname}/")
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


def load_sim(dirname: str):
    home = Path(__file__).parent
    save_dir = Path(home, f"results/{dirname}/")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(Path(save_dir, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)
    with open(Path(save_dir, "weights.npy"), "rb") as fd:
        weights = np.load(fd)
    with open(Path(save_dir, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(save_dir, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, weights, ground_energy
