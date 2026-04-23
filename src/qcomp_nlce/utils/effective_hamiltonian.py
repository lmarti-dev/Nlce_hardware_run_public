import numpy as np
from src.qcomp_nlce.utils import from_bitstring, hamming_weight_states
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import inv, sqrtm
from scipy.sparse.linalg import eigsh


def H_effective(
    gs_expectation: float, one_qp_expectation: np.ndarray, overlap: np.ndarray
) -> np.ndarray:
    N = one_qp_expectation.shape[0]
    B = overlap.conj().T @ inv(sqrtm(overlap @ overlap.conj().T))
    return B.conj().T @ one_qp_expectation @ B - np.eye(N) * gs_expectation


def H_effective_pcat(
    gs_expectation: float,
    one_qp_expectation: np.ndarray,
    overlap: np.ndarray,
    gs_0: complex,
    gs_1qp: np.ndarray,
    one_qp_row_0: np.ndarray,
) -> np.ndarray:
    N = one_qp_expectation.shape[0]
    A = overlap - np.outer(gs_1qp, one_qp_row_0) / gs_0
    T = A.conj().T @ inv(sqrtm(A @ A.conj().T))
    return T.conj().T @ one_qp_expectation @ T - np.eye(N) * gs_expectation


def calculate_Ed(hamiltonian: np.ndarray) -> float:
    eigvals, _ = eigsh(hamiltonian, k=1, which="SA")
    return float(np.real(eigvals[0]))


def H_effective_ed(hamiltonian: SparsePauliOp) -> np.ndarray:
    N = hamiltonian.num_qubits
    whole_ham = np.array(hamiltonian.to_matrix(False), dtype=complex)
    indices = np.array([2**j for j in range(N)], dtype=int)

    eigenvalues, eigenvectors = np.linalg.eigh(whole_ham)
    overlaps = np.sum(np.abs(eigenvectors[indices, :]) ** 2, axis=0)
    indices_eigenvectors = np.argsort(-overlaps)[:N]

    A = eigenvectors[np.ix_(indices, indices_eigenvectors)]
    B = np.diag(eigenvalues[indices_eigenvectors])

    T = A.conj().T @ inv(sqrtm(A @ A.conj().T))
    return T.conj().T @ B @ T - eigenvalues[0] * np.eye(N, dtype=complex)


def get_1qp_indices(ham_ref: SparsePauliOp, eigenvectors: np.ndarray) -> np.ndarray:
    N = ham_ref.num_qubits
    ref_mat = np.array(ham_ref.to_matrix(False), dtype=complex)
    _, eigvecs_ref = np.linalg.eigh(ref_mat)
    Q = eigvecs_ref[:, 1 : N + 1]
    proj = Q @ Q.conj().T
    column_norms = np.linalg.norm(proj @ eigenvectors, axis=0)
    return np.argsort(-column_norms)[:N]


def H_effective_ed_LF(hamiltonian: SparsePauliOp, ham_ref: SparsePauliOp) -> np.ndarray:
    N = hamiltonian.num_qubits
    whole_ham = np.array(hamiltonian.to_matrix(False), dtype=complex)
    indices = np.array([2**j for j in range(N)], dtype=int)

    eigenvalues, eigenvectors = np.linalg.eigh(whole_ham)
    indices_eigenvectors = get_1qp_indices(ham_ref, eigenvectors)

    v0 = eigenvectors[0, 0]
    gs = eigenvectors[:, 0:1]
    gs_row = eigenvectors[0:1, indices_eigenvectors]
    A = (eigenvectors[:, indices_eigenvectors] - (gs @ gs_row) / v0)[indices, :]
    B = np.diag(eigenvalues[indices_eigenvectors])

    T = A.conj().T @ inv(sqrtm(A @ A.conj().T))
    H_eff = T.conj().T @ B @ T - eigenvalues[0] * np.eye(N, dtype=complex)

    if np.allclose(H_eff.imag, 0.0, atol=1e-10):
        H_eff = H_eff.real
    return H_eff
