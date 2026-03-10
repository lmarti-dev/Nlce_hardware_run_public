import numpy as np
from qcomp_nlce.utils import from_bitstring, hamming_weight_states
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import inv, sqrtm
from scipy.sparse.linalg import eigsh

# def Expectation(Hamiltonian, states):
#     return states.conj().T @ Hamiltonian @ states


# def H_effective(hamiltonian, ind_1, gs_unitary, one_qp_unitary):

#     gs_estimate = Expectation(hamiltonian, gs_unitary)
#     H_HVA = Expectation(hamiltonian, one_qp_unitary)

#     one_qp_states = one_qp_unitary[ind_1, :]
#     B = one_qp_states[ind_1, :].conj().T * (one_qp_states[ind_1, :] * one_qp_states[ind_1, :].conj().T)^(-1/2)
#     H_effective = B.conj().T @ H_HVA @ B - (np.eye(len(ind_1)), len(ind_1), len(ind_1)) * gs_estimate

#     return H_effective


# def H_effective_pcat(hamiltonian, ind_1, gs_unitary, one_qp_unitary,overlap,  gs_expectation, one_qp_expectation):

#     N = one_qp_expectation.shape[0]
#     one_qp_states = overlap
#     Help_matrix = (gs_unitary/gs_unitary[0,0] * one_qp_states[0:0, :])

#     A = one_qp_states - Help_matrix
#     A = A[ind_1, :]
#     S_B = A
#     B = S_B.conj().conj().T @ (S_B @ S_B.conj().T)^(-1/2)

#     H_effective = B.conj().T * one_qp_expectation * B - np.eye(N) @ gs_expectation

#     return H_effective


def H_effective(
    gs_expectation: float, one_qp_expectation: np.ndarray, overlap: np.ndarray
) -> np.ndarray:

    # one qp expectation: <psi|U* H U |psi>
    # overlap: <psi|U|psi>
    #  gs expectation is ground state energy

    # Get dimension from one_qp_expectation matrix
    N = one_qp_expectation.shape[0]

    # Compute transformation matrix: B = (overlap @ overlap†)^(-1/2) @ overlap
    overlap_product = overlap @ overlap.conj().T
    sqrt_overlap_product = sqrtm(overlap_product)
    inv_sqrt = inv(sqrt_overlap_product)
    B = overlap.conj().T @ inv_sqrt
    # Apply PCAT transformation (Eq. 47)
    H_effective = B.conj().T @ one_qp_expectation @ B - np.eye(N) * gs_expectation

    return H_effective


def calculate_Ed(hamiltonian: np.ndarray):
    """
    Compute the smallest real eigenvalue of the Hamiltonian,
    equivalent to Julia eigs(... which=:SR).
    """
    # Convert to sparse matrix if needed
    try:
        # SciPy expects sparse matrix or ndarray
        H = hamiltonian
    except:
        H = np.array(hamiltonian, dtype=complex)

    # compute smallest eigenvalue (Hermitian)
    eigvals, _ = eigsh(H, k=1, which="SA")  # SA = smallest algebraic

    return float(np.real(eigvals[0]))


def H_effective_ed(hamiltonian: SparsePauliOp) -> np.ndarray:

    N = hamiltonian.num_qubits
    # make Hamiltonian a full dense matrix

    whole_ham = np.array(hamiltonian.to_matrix(False), dtype=complex)
    indices = np.array([2**j for j in range(N)], dtype=int)

    # diagonalize to get eigenvalues and eigenvectors
    # Julia's eigen(whole_ham) for Hermitian -> use eigh
    eigenvalues, eigenvectors = np.linalg.eigh(whole_ham)

    # calculate overlap of eigenvectors with unperturbed space of one spin flip
    # eigenvectors[indices, :] -> rows corresponding to those basis states
    # abs()^2 and sum over rows -> overlap for each eigenvector
    overlaps = np.sum(np.abs(eigenvectors[indices, :]) ** 2, axis=0)

    # find the indices of the N largest values of this overlap
    # equivalent to partialsortperm(..., rev=true)
    # here we fully sort; if you want exactly the partial behavior, use argpartition.
    indices_eigenvectors = np.argsort(overlaps)[-N:][::-1]

    # project proper eigenstates into one spin-flip state basis
    # A is N×N
    A = eigenvectors[np.ix_(indices, indices_eigenvectors)]

    # choose corresponding eigenvalues and put them on a diagonal matrix
    B = np.diag(eigenvalues[indices_eigenvectors])

    # calculate transformation
    aa_dag = A @ A.conj().T
    # matrix square root (not elementwise)
    sq_rt = sqrtm(aa_dag)
    T = A.conj().T @ inv(sq_rt)

    # transform Hamiltonian into effective Hamiltonian; subtract GS energy
    E0 = calculate_Ed(whole_ham)
    H_eff = T.conj().T @ B @ T - E0 * np.eye(N, dtype=complex)

    return H_eff
