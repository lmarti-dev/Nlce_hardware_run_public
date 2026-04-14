from qiskit.quantum_info import SparsePauliOp
import numpy as np
from fauplotstyle import style
from scipy.sparse.linalg import eigsh


def random_paulistring(n_qubits: int) -> SparsePauliOp:
    paulis = ["I", "X", "Y", "Z"]
    ps = np.random.choice(paulis, size=n_qubits, replace=True)
    return SparsePauliOp("".join(ps))


def random_ham(n_qubits: int, n_terms: int) -> tuple[SparsePauliOp, list[float]]:
    y = 0
    coeffs = []
    for n in range(n_terms):
        coeff = 2 * (np.random.rand() - 1)
        y += coeff * random_paulistring(n_qubits)
        coeffs.append(coeff)
    return y, coeffs


if __name__ == "__main__":
    n_samples = 100
    emaxes = []
    coeffsums = []
    n_qubits = 5
    n_terms = n_qubits**2
    for n in range(n_samples):
        print(f"{n}/{n_samples}", end="\r")
        ham, coeffs = random_ham(n_qubits, n_terms)
        emax, _ = eigsh(ham.to_matrix(True), k=1, which="LA")
        emaxes.append(emax)

        coeffsums.append(np.sum(np.abs(coeffs)))

    print(
        f"emax: {np.mean(emaxes)}, {np.std(emaxes)}, cf: {np.mean(coeffsums)}, {np.std(coeffsums)}"
    )
