from qiskit.quantum_info import SparsePauliOp, Pauli
from scipy.sparse.linalg import eigsh


def get_op_indices(op: SparsePauliOp) -> list:
    all_indices = []
    for term in op:
        p = term.paulis[0]
        all_indices.append(get_pauli_indices(p))
    return all_indices


def get_pauli_indices(term: Pauli) -> list:
    return [ind for ind in range(len(term)) if str(term)[ind] != "I"]


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
            for s in range(n_qubits)
        ]
    )

    hamiltonian = hamiltonian_J + hamiltonian_g

    return hamiltonian


def get_ground_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    ground_energy, _ = eigsh(mat, k=1, which="SA")
    return ground_energy


def pretty_pauli_str(p: Pauli) -> str:
    idx = get_pauli_indices(p)
    return "".join([f"{x}{y}" for x, y in zip(p[::-1][idx], idx)])
