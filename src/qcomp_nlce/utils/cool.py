from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
import numpy as np
import itertools


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
    coupler_type: str = "XX+YY",
) -> QuantumCircuit:
    n_qubits = ham.num_qubits

    alpha = omega / (weaken_coupling * n_qubits)
    anc_ham = 0.5 * (
        SparsePauliOp("I" * (n_qubits + 1)) - SparsePauliOp("I" * n_qubits + "Z")
    )
    circuit = QuantumCircuit(ham.num_qubits + 1)
    if coupler_type == "XX+YY":
        coupler = SparsePauliOp("I" * (n_qubits - 1) + "XX") + SparsePauliOp(
            "I" * (n_qubits - 1) + "YY"
        )
    elif coupler_type == "proj":
        w, v = np.linalg.eigh(ham.to_matrix())
        proj = np.multiply.outer(v[:, 0], np.conj(v[:, 1]))
        coupler = mat_to_pauli(proj)
        coupler = (SparsePauliOp("X") - 1j * SparsePauliOp("Y")).expand(coupler)
        coupler += coupler.adjoint()
        coupler = coupler.simplify()
    evolution_time = np.pi / alpha
    circuit.append(
        PauliEvolutionGate(
            SparsePauliOp("I").expand(ham) + coupler * alpha + omega * anc_ham,
            evolution_time,
        ),
        range(ham.num_qubits + 1),
    )
    return circuit
