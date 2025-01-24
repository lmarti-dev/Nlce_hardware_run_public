from qiskit.quantum_info import SparsePauliOp
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


def get_ground_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    ground_energy, _ = eigsh(mat, k=1, which="SA")
    return ground_energy
