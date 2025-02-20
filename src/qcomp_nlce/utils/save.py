from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import SparsePauliOp

import os
from pathlib import Path
import numpy as np


def get_save_directory(dirname: str):
    home = Path(__file__).parent.parent
    save_dir = Path(home, f"res_sim/{dirname}/")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save_sweep(
    dirname: str,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    ground_energy: float,
):
    save_dir = get_save_directory(dirname)
    with open(Path(save_dir, "circuit.qpy"), "wb+") as fd:
        qpy.dump(circuit, fd)
    with open(Path(save_dir, "ground_energy.npy"), "wb+") as fd:
        np.save(fd, np.array(ground_energy))
    with open(Path(save_dir, "hamiltonian.npy"), "wb+") as fd:
        np.save(fd, hamiltonian.to_list(True))


def load_sweep(dirname: str):
    save_dir = get_save_directory(dirname)
    with open(Path(save_dir, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)[0]
    with open(Path(save_dir, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(save_dir, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, ground_energy


def save_optimized_vqe(
    dirname: str,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    weights: np.ndarray,
    ground_energy: float,
):
    save_dir = get_save_directory(dirname)
    with open(Path(save_dir, "circuit.qpy"), "wb+") as fd:
        qpy.dump(circuit, fd)
    with open(Path(save_dir, "weights.npy"), "wb+") as fd:
        np.save(fd, weights)
    with open(Path(save_dir, "ground_energy.npy"), "wb+") as fd:
        np.save(fd, np.array(ground_energy))
    with open(Path(save_dir, "hamiltonian.npy"), "wb+") as fd:
        np.save(fd, hamiltonian.to_list(True))


def load_optimized_vqe(dirname: str):
    save_dir = get_save_directory(dirname)
    with open(Path(save_dir, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)[0]
    with open(Path(save_dir, "weights.npy"), "rb") as fd:
        weights = np.load(fd)
    with open(Path(save_dir, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(save_dir, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, weights, ground_energy
