from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.qcomp_nlce.utils import (
    iterative_hadamard_test_estimator,
    get_appropriate_fake_backend,
    custom_idx_groups_of_n,
)


def main(n_shots: int):
    circ = QuantumCircuit(1)
    circ.h(0)
    circ.s(0)
    circ.h(0)
    circ.rx(0.1, 0)
    circ.rx(0.1, 0)
    circ.rx(0.1, 0)
    circ.rx(0.1, 0)
    circ.rx(0.1, 0)
    circ.rx(0.1, 0)
    circ.rx(0.1, 0)
    circ.rx(0.1, 0)

    sv = Statevector(circ).data
    true_val = sv[0]

    custom_idx = custom_idx_groups_of_n(len([c for c in circ]), 3)
    est_sv_g = iterative_hadamard_test_estimator(
        circ, parameters=None, n_shots=n_shots, custom_idx=custom_idx
    )

    fake_backend = get_appropriate_fake_backend(circ.num_qubits, which="IBM")
    est_fake_g = iterative_hadamard_test_estimator(
        circ, parameters=None, n_shots=n_shots, custom_idx=None, backend=fake_backend
    )

    print(f"true: {true_val:.4f} sv: {est_sv_g:.3f} fake: {est_fake_g:.3f}")


if __name__ == "__main__":
    main(int(1e6))
