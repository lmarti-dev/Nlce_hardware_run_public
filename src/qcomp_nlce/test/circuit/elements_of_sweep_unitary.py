from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.qcomp_nlce.utils import (
    get_tfim_sweep_circuit,
    TFIM,
    get_min_gap,
    circuit_to_matrix,
)


def diagonalize_ham(ham: SparsePauliOp):
    mat = ham.to_matrix()
    t = np.linalg.eigh(mat)
    return t.eigenvalues, t.eigenvectors


def eigen_decomposition(v: np.ndarray, vecs: np.ndarray):
    return np.abs(np.conj(vecs.T) @ v)


def get_1qp_mat(circuit: QuantumCircuit):
    mat = circuit_to_matrix(circuit)
    n = int(np.log2(mat.shape[0]))
    sl = [int(2**x) for x in range(n)]
    return mat[np.ix_(sl, sl)]


def plot_eigvecs_map(eig_vecs: np.ndarray):
    fig, ax = plt.subplots()
    _n = eig_vecs.shape[0]
    y, x = np.meshgrid(range(_n), range(_n))

    pcm0 = ax.pcolormesh(y, x, np.abs(eig_vecs), norm=LogNorm())
    fig.colorbar(pcm0, ax=ax, extend="max")

    plt.show()


if __name__ == "__main__":
    n_qubits = 10

    ham_j, ham_h = TFIM(n_qubits, 1, 1, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = 1 / (min_gap**2)
    n_steps = 10 * hamiltonian.num_qubits
    mult = 1

    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=(mult * total_time),
        n_steps=int(mult * n_steps),
    )

    eig_vals, eig_vecs = diagonalize_ham(hamiltonian)

    plot_eigvecs_map(eig_vecs)

    results = np.zeros((2**n_qubits, n_qubits))

    u_matrix = circuit_to_matrix(circuit)

    u_1qp = get_1qp_mat(u_matrix)

    plot_eigvecs_map(u_matrix)

    print(u_1qp)
