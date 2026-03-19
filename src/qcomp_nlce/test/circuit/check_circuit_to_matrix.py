from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit
import numpy as np

import matplotlib.pyplot as plt

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


def prep_1qp(circuit: QuantumCircuit, qnum: int) -> QuantumCircuit:
    pcirc = QuantumCircuit(circuit.qubits)
    pcirc.x(qnum)
    pcirc.compose(circuit, inplace=True)
    return pcirc


if __name__ == "__main__":
    n_qubits = 6

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

    psi = np.zeros((2**n_qubits,), dtype=complex)
    psi[0] = 1

    fig, ax = plt.subplots()

    for q in range(n_qubits):
        circ_in = QuantumCircuit(circuit.qubits)
        circ_in.x(q)
        circ_out = circ_in.compose(circuit)
        v = Statevector(circ_out).__array__()
        mat = circuit_to_matrix(circ_out)
        out = mat @ psi
        print(f"circ-mat fid: {np.abs(np.dot(v.conj(),out))}")
        fids = eigen_decomposition(out, eig_vecs)
        out_comps = np.where(fids > 1e-1)[0]
        print(f"{q}: {out_comps}", len(out_comps))
        ax.plot(range(len(fids)), fids)
    ax.set_yscale("log")
    ax.set_ylim([1e-2, 1])

    plt.show()
