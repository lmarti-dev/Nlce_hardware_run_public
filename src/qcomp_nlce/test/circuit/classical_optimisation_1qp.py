from qcomp_nlce.utils import TFIM
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


def pauli_on_idx(idx: int, n_qubits: int, pauli: str = "X"):
    s = "I" * idx + pauli + "I" * (n_qubits - 1 - idx)
    return SparsePauliOp(s)


def identity_coeff(hamiltonian: SparsePauliOp):
    identity = SparsePauliOp("I" * hamiltonian.num_qubits)
    for term in hamiltonian:
        if term.paulis == identity.paulis:
            return term.coeffs
    return 0


def setup_ansatz(hamiltonian: SparsePauliOp) -> callable:
    n_qubits = hamiltonian.num_qubits
    mat = np.zeros((n_qubits, n_qubits), dtype=complex)
    for x_j in range(n_qubits):
        for x_k in range(n_qubits):
            out = (
                pauli_on_idx(x_j, n_qubits) @ hamiltonian @ pauli_on_idx(x_k, n_qubits)
            )
            coeff = identity_coeff(out)
            if coeff != 0:
                mat[x_j, x_k] = coeff[0]

    def ansatz(xk: np.ndarray) -> float:
        c = xk[:n_qubits] + 1j * xk[n_qubits:]
        c /= np.linalg.norm(c)
        return np.real(c.conjugate().dot(mat @ c))

    return ansatz


def optimize_ansatz(hamiltonian: SparsePauliOp) -> list:
    energy_values = []
    xks = []

    ansatz = setup_ansatz(hamiltonian)

    def callback(xk: np.ndarray, *args):
        e = fun(xk=xk)
        print(e, end="\r")
        energy_values.append(e)
        xks.append(xk)

    def fun(xk: np.ndarray):
        return ansatz(xk)

    minimize(
        fun=fun,
        x0=np.random.rand(2 * n_qubits),
        method="CG",
        callback=callback,
    )
    return energy_values, xks


if __name__ == "__main__":

    n_qubits = 9
    h = 1
    J = 1
    periodic = False

    hamiltonian = TFIM(n_qubits, h, J, periodic=periodic)

    energy_values, xks = optimize_ansatz(hamiltonian)
