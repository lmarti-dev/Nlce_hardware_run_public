from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit
import numpy as np

import matplotlib.pyplot as plt

from qcomp_nlce.utils import (
    get_tfim_sweep_circuit,
    TFIM,
    get_min_gap,
)


def diagonalize_ham(ham: SparsePauliOp):
    mat = ham.to_matrix()
    t = np.linalg.eigh(mat)
    return t.eigenvalues, t.eigenvectors


def eigen_decomposition(v: np.ndarray, vecs: np.ndarray):
    return np.abs(np.conj(vecs.T) @ v)


def prep_1qp(circuit: QuantumCircuit, qnum: int) -> QuantumCircuit:
    pcirc = QuantumCircuit(circuit.qubits)
    pcirc.x(qnum)
    pcirc.compose(circuit, inplace=True)
    return pcirc


if __name__ == "__main__":
    n_qubits = 10

    ham_j, ham_h = TFIM(n_qubits, 1, 0.4, periodic=False, split_parts=True)
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

    results = np.zeros((2**n_qubits, n_qubits))

    for qnum in range(n_qubits):
        pcirc = prep_1qp(circuit, qnum)
        v = Statevector(pcirc)
        r = eigen_decomposition(v.__array__(), eig_vecs)
        results[:, qnum] = r

    fig, ax = plt.subplots()

    for n in range(n_qubits):
        ax.plot(range(2**n_qubits), results[:, n], label=f"{n}")
        out_comps = np.where(results[:, n] > 1e-1)[0]
        print(f"{n}: {out_comps}", len(out_comps))
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim([1e-1, 1])

    # plt.show()
