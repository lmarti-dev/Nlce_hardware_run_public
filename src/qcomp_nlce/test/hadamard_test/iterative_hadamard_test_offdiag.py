from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qcomp_nlce.utils import (
    iterative_hadamard_test_off_diag,
    get_appropriate_fake_backend,
    custom_idx_groups_of_n,
    circuit_to_matrix,
)
import numpy as np


def assigned_wrt_diag(
    x: str, y: str, mat: np.ndarray, jUk: complex, kUj: complex
) -> np.ndarray:
    if int(x) != int(y):
        mat[int(x), int(y)] = jUk
        mat[int(y), int(x)] = kUj
    else:
        mat[int(x), int(y)] = jUk
    return mat


def main(n_shots: int):
    circ = QuantumCircuit(1)
    circ.rx(0.4, 0)
    circ.h(0)
    circ.s(0)
    true_comp_vals = circuit_to_matrix(circ)

    sv_comp_vals = np.zeros((2**circ.num_qubits, 2**circ.num_qubits), dtype=complex)
    fake_comp_vals = np.zeros((2**circ.num_qubits, 2**circ.num_qubits), dtype=complex)

    bs = ("0", "1")
    for jj, x in enumerate(bs):
        for y in bs[jj:]:
            custom_idx = custom_idx_groups_of_n(len([c for c in circ]), 1)
            jUk, kUj = iterative_hadamard_test_off_diag(
                circ, None, n_shots, x, y, custom_idx=custom_idx
            )

            sv_comp_vals = assigned_wrt_diag(x, y, sv_comp_vals, jUk, kUj)

            # fake_backend = get_appropriate_fake_backend(circ.num_qubits, which="IBM")
            # fjUk, fkUj = iterative_hadamard_test_off_diag(
            #     circ, None, n_shots, x, y, custom_idx=custom_idx, backend=fake_backend
            # )

            fjUk, fkUj = (0, 0)

            fake_comp_vals = assigned_wrt_diag(x, y, fake_comp_vals, fjUk, fkUj)

    print(np.round(true_comp_vals, 3))
    print(np.round(sv_comp_vals, 3))
    # print(np.round(fake_comp_vals, 3))

    print("done")


if __name__ == "__main__":
    main(int(1e6))
