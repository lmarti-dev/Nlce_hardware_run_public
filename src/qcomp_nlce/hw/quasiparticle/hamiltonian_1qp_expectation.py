from pathlib import Path

import numpy as np
from src.qcomp_nlce.utils import (
    circ_ham_expectation_values,
    compare_with_true,
    get_appropriate_fake_backend,
    get_backend,
    hamming_weight_states,
    measure_hamiltonian_elements_v2,
    BACKEND_TYPES,
    RUNS_POSTFIX,
    HAM_EXPECT_METHODS,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


HOME = Path(__file__).parent


def loop_measure_1qp_expectations_v2(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    which_backends: list[str],
    solve: bool = True,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
    n_reps: int = 1,
    which_algorithm: HAM_EXPECT_METHODS = "estimator",
    postfix: RUNS_POSTFIX = "",
    mitigate_depolarization: bool = False,
):
    n_qubits = circuit.num_qubits
    comp_states = hamming_weight_states(n_qubits, 1)
    if solve:
        true_comp_vals = circ_ham_expectation_values(hamiltonian, circuit, comp_states)
    else:
        true_comp_vals = None
    d = {
        "system": {"n_qubits": circuit.num_qubits, "values": true_comp_vals},
    }
    for which_backend in which_backends:
        if which_backend == "statevector":
            n_shots *= n_reps
        print(f"==== {which_backend} ====")
        item = measure_1qp_expectations_v2(
            hamiltonian,
            circuit,
            which_backend,
            true_comp_vals,
            provider_name,
            n_shots,
            n_reps=n_reps,
            which_algorithm=which_algorithm,
            postfix=postfix,
            mitigate_depolarization=mitigate_depolarization,
        )
        d[which_backend] = item[which_backend]
    return d


def measure_1qp_expectations_v2(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    which_backend: BACKEND_TYPES,
    true_comp_vals: np.ndarray = None,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
    n_reps: int = 1,
    which_algorithm: HAM_EXPECT_METHODS = "estimator",
    postfix: RUNS_POSTFIX = "",
    mitigate_depolarization: bool = False,
) -> np.ndarray:
    n_qubits = circuit.num_qubits
    print(
        f"Minimal trace error (only shot noise) {0.5 * (n_qubits) / np.sqrt(n_shots):.4e}"
    )

    comp_states = hamming_weight_states(n_qubits, 1)

    if which_backend == "statevector":
        bn = which_backend
        backend = which_backend

    elif which_backend == "fake_backend" or which_backend == "real_backend":
        if which_backend == "fake_backend":
            backend = get_appropriate_fake_backend(
                circuit.num_qubits, which=provider_name
            )
        elif which_backend == "real_backend":
            backend = get_backend(circuit.num_qubits, provider_name)
        bn = backend.name

    comp_vals, stds = measure_hamiltonian_elements_v2(
        hamiltonian,
        circuit,
        comp_states,
        None,
        backend,
        which_backend,
        n_shots,
        n_reps=n_reps,
        which_algorithm=which_algorithm,
        postfix=postfix,
        mitigate_depolarization=mitigate_depolarization,
    )
    print(f"{which_backend} ({bn})")
    td = compare_with_true(true_comp_vals, comp_vals, bn)

    return {
        which_backend: {
            "name": bn,
            "trace_dist": td,
            "values": comp_vals,
            "stds": stds,
        }
    }
