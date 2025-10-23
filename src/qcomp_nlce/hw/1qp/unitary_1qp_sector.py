from qiskit import QuantumCircuit
import numpy as np
from qcomp_nlce.utils import (
    circuit_matrix_elements,
    measure_matrix_elements,
    get_appropriate_fake_backend,
    get_backend,
    get_min_gap,
    TFIM,
    hamming_weight_states,
    get_tfim_sweep_circuit,
    measure_matrix_elements_merge_circuits,
    load_optimized_vqe,
    apply_parameters,
    get_tfim_ratio,
)
import os
from pathlib import Path

HOME = Path(__file__).parent


def trace_dist(u1: np.ndarray, u2: np.ndarray) -> float:
    diff = u1 - u2
    out = np.sqrt(diff.T.conj() @ diff)
    return 0.5 * np.real(np.trace(out))


def measure_1qp_elements(
    circuit: QuantumCircuit,
    provider: str = "IBM",
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
) -> np.ndarray:

    mme_func = measure_matrix_elements_merge_circuits

    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")
    comp_states = hamming_weight_states(n_qubits, 1)
    true_comp_vals = circuit_matrix_elements(circuit, comp_states)

    # the matrix you get otherwise is indexed backwards wrt to what you get
    # from the circuit-to-matrix qiskit function
    true_comp_vals = true_comp_vals[::-1, ::-1]

    sv_comp_vals = mme_func(
        circuit, comp_states, circuit.parameters, "statevector", n_shots
    )
    sv_fidelity = trace_dist(true_comp_vals, sv_comp_vals)
    print(f"Statevector {sv_fidelity:.5e}")

    backend_name = "AQT" if provider == "LRZ" else "IBM"
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, which=backend_name)
    fake_comp_vals = mme_func(
        circuit, comp_states, circuit.parameters, fake_backend, n_shots
    )
    fake_fidelity = trace_dist(true_comp_vals, fake_comp_vals)
    print(f"Fake Backend ({fake_backend.name}) {fake_fidelity:.5e}")

    if run_on_hardware:
        real_backend = get_backend(circuit.num_qubits, backend_name)
        real_comp_vals = mme_func(
            circuit, comp_states, circuit.parameters, real_backend, n_shots
        )
        real_fidelity = trace_dist(true_comp_vals, real_comp_vals)
        print(f"RealBackend ({real_backend.name}) {real_fidelity:.5e}")


def measure_sweep_1qp(n_shots=10000):
    n_qubits = 4
    ham_j, ham_h = TFIM(n_qubits, 1, 1, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = 1 / (min_gap**2)
    n_steps = 1 * hamiltonian.num_qubits
    mult = 1

    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=(mult * total_time),
        n_steps=int(mult * n_steps),
    )

    print(f"{hamiltonian.num_qubits} qubits, J/h {get_tfim_ratio(hamiltonian)}")
    print(f"sweep depth: {circuit.depth()}")

    measure_1qp_elements(circuit, n_shots=n_shots)


def measure_vqe_1qp(n_shots: int = 10000):
    dirpath = Path(HOME, "../../res_sim/1qp_small/vqe/")
    for dirname in os.listdir(dirpath):
        hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(
            Path(dirpath, dirname)
        )
        print(f"{hamiltonian.num_qubits} qubits, J/h {get_tfim_ratio(hamiltonian)}")
        circuit_with_params = apply_parameters(circuit, weights[-1])
        measure_1qp_elements(circuit_with_params, n_shots=n_shots)


if __name__ == "__main__":
    n_shots = int(1e7)
    print(f"n_shots: {n_shots}")
    print("=== Sweep ===")
    measure_sweep_1qp(n_shots)
    print("=== VQE ===")
    measure_vqe_1qp(n_shots)
