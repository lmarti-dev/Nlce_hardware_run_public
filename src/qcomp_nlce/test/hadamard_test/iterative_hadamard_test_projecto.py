from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qcomp_nlce.utils import (
    iterative_hadamard_test_projecto_estimo,
    get_appropriate_fake_backend,
    custom_idx_groups_of_n,
    circuit_to_matrix,
)
import numpy as np


def main(n_shots: int):
    circ = QuantumCircuit(1)
    circ.h(0)
    circ.s(0)
    circ.rx(1, 0)
    true_comp_vals = circuit_to_matrix(circ)
    print(np.round(true_comp_vals, 3))

    for zz in ("0", "1"):
        x = zz
        y = zz
        custom_idx = custom_idx_groups_of_n(len([c for c in circ]), 1)
        est_sv_g = iterative_hadamard_test_projecto_estimo(
            circ, None, n_shots, x, y, custom_idx=custom_idx
        )

        fake_backend = get_appropriate_fake_backend(circ.num_qubits, which="IBM")
        est_fake_g = iterative_hadamard_test_projecto_estimo(
            circ, None, n_shots, x, y, custom_idx=custom_idx, backend=fake_backend
        )

        print(
            f"true: {true_comp_vals[int(x),int(y)]:.4f} sv: {est_sv_g:.3f} fake: {est_fake_g:.3f}"
        )


if __name__ == "__main__":
    main(int(1e6))
