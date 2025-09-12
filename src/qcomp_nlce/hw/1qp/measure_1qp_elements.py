from qiskit import QuantumCircuit
import numpy as np
from qcomp_nlce.utils import (
    get_1qp_mat,
    measure_matrix_elements,
    get_appropriate_fake_backend,
    get_backend,
    get_min_gap,
    TFIM,
    hamming_weight_states,
    get_tfim_sweep_circuit,
)


def measure_1qp_elements(
    circuit: QuantumCircuit,
    provider: str = "IBM",
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    get_true_values: bool = True,
) -> np.ndarray:
    n_qubits = circuit.num_qubits
    if get_true_values:
        true_comp_vals = get_1qp_mat(circuit)

    comp_states = hamming_weight_states(n_qubits, 1)
    backend_name = "AQT" if provider == "LRZ" else "IBM"
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, which=backend_name)
    fake_comp_vals = measure_matrix_elements(
        circuit, comp_states, circuit.parameters, fake_backend, n_shots
    )

    if run_on_hardware:
        real_backend = get_backend(circuit.num_qubits, backend_name)
        real_comp_vals = measure_matrix_elements(
            circuit, comp_states, circuit.parameters, real_backend, n_shots
        )
    if get_true_values:
        fake_fidelity = np.trace(np.abs(true_comp_vals - fake_comp_vals))
        print(f"Fake Backend ({fake_backend.name}) {fake_fidelity:.5e}  ")
        if run_on_hardware:
            real_fidelity = np.trace(np.abs(true_comp_vals - real_comp_vals))
            print(f"RealBackend ({real_backend.name}) {real_fidelity:.5e}")


def measure_sweep_1qp():
    n_qubits = 3
    ham_j, ham_h = TFIM(n_qubits, 1, 1, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = 0.2 / (min_gap**2)
    n_steps = 1 * hamiltonian.num_qubits
    mult = 1

    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=(mult * total_time),
        n_steps=int(mult * n_steps),
    )

    measure_1qp_elements(
        circuit,
    )


if __name__ == "__main__":
    measure_sweep_1qp()
