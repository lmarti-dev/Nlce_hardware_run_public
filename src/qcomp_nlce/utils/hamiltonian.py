from qiskit.quantum_info import SparsePauliOp, Pauli
from scipy.sparse.linalg import eigsh
import numpy as np
from typing import Union


def binleftpad(integer: int, left_pad: int):
    b = format(integer, "0{lp}b".format(lp=left_pad))
    return b


def to_bitstring(ind: int, n_qubits: int, right_to_left: bool = False) -> str:
    if isinstance(ind, (int, np.integer)):
        if n_qubits is None:
            b = bin(ind)
        else:
            b = binleftpad(integer=ind, left_pad=n_qubits)
    elif isinstance(ind, str):
        b = ind
    else:
        raise ValueError(f"Expected ind to be int or str, got: {type(ind)}")
    if "b" in b:
        b = b.split("b")[1]
    if right_to_left:
        b = "".join(list(reversed(b)))

    return b


def index_bits(
    a: Union[str, int], N: int = None, ones=True, right_to_left: bool = False
) -> list:
    """Takes a binary number and returns a list of indices where the bit is one (or zero)
    Args:
        a (binary number): The binary number whose ones or zeroes will be indexed
        ones (bool): If true, index ones. If false, index zeroes
    Returns:
        list: List of indices where a is one (or zero)
    """
    if isinstance(a, int):
        if N is None:
            b = bin(a)
        else:
            b = binleftpad(integer=a, left_pad=N)
    else:
        b = a
    if "b" in b:
        b = b.split("b")[1]
    if right_to_left:
        b = list(reversed(b))
    if ones:
        return [idx for idx, v in enumerate(b) if int(v)]
    elif not ones:
        return [idx for idx, v in enumerate(b) if not int(v)]


def from_bitstring(b: str, n_qubits: int, right_to_left: bool = False) -> int:
    if "b" in b:
        b = b.split("b")[1]
    idx = index_bits(a=b, N=n_qubits, right_to_left=True)
    if right_to_left:
        return sum([2 ** (n_qubits - 1 - iii) for iii in idx])
    else:
        return sum([2**iii for iii in idx])


def get_op_indices(op: SparsePauliOp) -> list:
    all_indices = []
    for term in op:
        p = term.paulis[0]
        all_indices.append(get_pauli_indices(p))
    return all_indices


def get_pauli_indices(term: Pauli) -> list:
    return [ind for ind in range(len(term)) if str(term)[ind] != "I"]


def get_singlet_ham(n_qubits: int, h: float, J: float) -> SparsePauliOp:
    hamiltonian = sum(
        [
            -J * SparsePauliOp(s * "I" + "XX" + (n_qubits - 2 - s) * "I")
            for s in range(0, n_qubits - 1, 2)
        ]
    )
    return hamiltonian


def get_tfim_ratio(ham: SparsePauliOp) -> float:
    c = ham.coeffs
    u = np.unique_counts(c)
    idx = np.argsort(u.counts)
    j = u.values[idx[0]]
    h = u.values[idx[-1]]
    return np.real(j / h).astype(float)


def get_tfim_ham(
    n_qubits: int, h: float, J: float, periodic: bool = False, split_parts: bool = False
) -> SparsePauliOp:

    hamiltonian_J = sum(
        [
            -J * SparsePauliOp(s * "I" + "XX" + (n_qubits - 2 - s) * "I")
            for s in range(n_qubits - 1)
        ]
    )
    if periodic:
        hamiltonian_J += SparsePauliOp("X" + "I" * (n_qubits - 2) + "X")
    hamiltonian_h = sum(
        [
            -h * SparsePauliOp(s * "I" + "Z" + (n_qubits - 1 - s) * "I")
            for s in range(n_qubits)
        ]
    )

    if split_parts:
        # J: XX h: Z
        return hamiltonian_J, hamiltonian_h

    hamiltonian = hamiltonian_J + hamiltonian_h

    return hamiltonian


def get_ground_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    ground_energy, _ = eigsh(mat, k=1, which="SA")
    return ground_energy[0]


def get_min_gap(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    min_gap = 0
    k = 2
    while np.isclose(min_gap, 0):
        energies, _ = eigsh(mat, k=k, which="SA")
        min_gap = np.abs(energies[-1] - energies[0])
        k += 1
    return min_gap


def pretty_pauli_str(p: Pauli) -> str:
    idx = get_pauli_indices(p)
    return "".join([f"{x}{y}" for x, y in zip(p[::-1][idx], idx)])


def subspace_diagonalize(
    unique_bitstrings: Union[list[int], list[str]], ham: SparsePauliOp
):
    if isinstance(unique_bitstrings[0], str):
        ind_list = [
            from_bitstring(bx, len(bx), right_to_left=True) for bx in unique_bitstrings
        ]
    else:
        ind_list = [int(x) for x in unique_bitstrings]

    sparse_ham = ham.to_matrix(sparse=True)
    n_samples = len(unique_bitstrings)
    sub_ham = np.zeros((n_samples, n_samples), dtype=complex)
    for x in range(n_samples):
        for y in range(x, len(unique_bitstrings)):
            ind_x = ind_list[x]
            ind_y = ind_list[y]
            sub_ham[x, y] = sparse_ham[ind_x, ind_y]
    sub_ham += np.triu(sub_ham, k=1).T
    ground_energy = np.min(np.linalg.eigvalsh(sub_ham))
    return ground_energy
