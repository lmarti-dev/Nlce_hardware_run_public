from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit import QuantumCircuit
import numpy as np
import itertools

from scipy.sparse.linalg import eigsh


def mat_to_pauli(mat: np.ndarray) -> SparsePauliOp:
    out = []
    r = int(np.log2(len(mat)))
    combs = itertools.product(*("IXYZ",) * r)
    for comb in combs:
        pauli = SparsePauliOp("".join(comb))
        pauli_mat = pauli.to_matrix()
        c = np.trace(mat @ pauli_mat)
        if not np.isclose(np.abs(c), 0):
            out.append(c * pauli)
    return sum(out)


def cooling_step(
    ham: SparsePauliOp,
    omega: float,
    weaken_coupling: float,
    n_steps: int,
    coupler_type: str = "XX+YY",
) -> QuantumCircuit:
    n_qubits = ham.num_qubits

    alpha = omega / (weaken_coupling * n_qubits)
    anc_ham = 0.5 * (
        SparsePauliOp("I" * (n_qubits + 1)) - SparsePauliOp("I" * n_qubits + "Z")
    )
    circuit = QuantumCircuit(ham.num_qubits + 1)
    coupler = get_coupler(ham, coupler_type, n_qubits)
    evolution_time = np.pi / alpha
    cooling_ham = SparsePauliOp("I").expand(ham) + coupler * alpha + omega * anc_ham
    for step in range(n_steps):
        circuit.append(
            PauliEvolutionGate(
                cooling_ham,
                evolution_time / n_steps,
            ),
            range(ham.num_qubits + 1),
        )
    return circuit


def get_coupler(ham: SparsePauliOp, coupler_type: str, n_qubits: int) -> SparsePauliOp:
    if coupler_type == "XX+YY":
        coupler = SparsePauliOp("I" * (n_qubits - 1) + "XX") + SparsePauliOp(
            "I" * (n_qubits - 1) + "YY"
        )
    elif coupler_type == "XX":
        coupler = SparsePauliOp("I" * (n_qubits - 1) + "XX")
    elif coupler_type == "proj":
        to_pauli = False
        w, v = eigsh(ham.to_matrix(True), k=2, which="SA")
        proj = np.multiply.outer(v[:, 0], np.conj(v[:, 1]))
        if to_pauli:
            coupler = mat_to_pauli(proj)
            coupler = (SparsePauliOp("X") - 1j * SparsePauliOp("Y")).expand(coupler)
            coupler += coupler.adjoint()
            coupler = coupler.simplify()
        else:
            raise NotImplementedError("Cannot make a Hamiltonian without that")

    return coupler
