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
    load_optimized_vqe,
    apply_parameters,
    get_tfim_ratio,
    custom_idx_groups_of_n,
    trace_dist,
    compare_with_true,
    HADAMARD_TESTS,
)
import os
from pathlib import Path

from data_manager import ExperimentDataManager
from qiskit_ibm_runtime import Session

HOME = Path(__file__).parent


def loop_measure_1qp_elements_v2(
    circuit: QuantumCircuit,
    which_backends: list[str],
    solve: bool = True,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
    group_size: int = None,
    which_hadamard_test: str = "full",
):
    n_qubits = circuit.num_qubits
    comp_states = hamming_weight_states(n_qubits, 1)
    if solve:
        true_comp_vals = circuit_matrix_elements(circuit, comp_states)
    else:
        true_comp_vals = None
    d = {
        "system": {"n_qubits": circuit.num_qubits, "values": true_comp_vals},
    }
    for which_backend in which_backends:
        print(f"==== {which_backend} ====")
        item = measure_1qp_elements_v2(
            circuit,
            which_backend,
            true_comp_vals,
            provider_name,
            n_shots,
            group_size,
            which_hadamard_test,
        )
        d[which_backend] = item[which_backend]
    return d


def measure_1qp_elements_v2(
    circuit: QuantumCircuit,
    which_backend: str,
    true_comp_vals: np.ndarray = None,
    provider_name: str = "IBM",
    n_shots: int = 1e3,
    group_size: int = None,
    which_hadamard_test: HADAMARD_TESTS = "full",
):

    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")
    comp_states = hamming_weight_states(n_qubits, 1)
    true_comp_vals = circuit_matrix_elements(circuit, comp_states)

    # the matrix you get otherwise is indexed backwards wrt to what you get
    # from the circuit-to-matrix qiskit function
    true_comp_vals = true_comp_vals[::-1, ::-1]

    if which_hadamard_test != "full":
        if group_size is None:
            group_size = len(circuit)
        custom_idx = custom_idx_groups_of_n(len(circuit), group_size)
    else:
        custom_idx = None

    if which_backend == "statevector":
        bn = which_backend
        backend = bn

    elif which_backend == "fake_backend" or which_backend == "real_backend":

        if which_backend == "fake_backend":
            backend = get_appropriate_fake_backend(
                circuit.num_qubits, which=provider_name
            )
        elif which_backend == "real_backend":
            backend = get_backend(circuit.num_qubits, provider_name)
        bn = backend.name

    if which_backend == "real_backend":
        with Session(backend) as s:
            comp_vals, stds = measure_matrix_elements(
                circuit,
                comp_states,
                circuit.parameters,
                backend,
                which_backend,
                n_shots,
                custom_idx=custom_idx,
                which_hadamard_test=which_hadamard_test,
            )
    else:
        comp_vals, stds = measure_matrix_elements(
            circuit,
            comp_states,
            circuit.parameters,
            backend,
            which_backend,
            n_shots,
            custom_idx=custom_idx,
            which_hadamard_test=which_hadamard_test,
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


def measure_1qp_elements(
    circuit: QuantumCircuit,
    backend_name: str = "IBM",
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    group_size: int = None,
    which: str = "full",
) -> np.ndarray:

    # DEPRECATED
    # DEPRECATED
    # DEPRECATED
    # DEPRECATED
    # DEPRECATED
    # DEPRECATED
    # DEPRECATED
    # DEPRECATED

    mme_func = measure_matrix_elements

    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")
    comp_states = hamming_weight_states(n_qubits, 1)
    true_comp_vals = circuit_matrix_elements(circuit, comp_states)

    # the matrix you get otherwise is indexed backwards wrt to what you get
    # from the circuit-to-matrix qiskit function
    true_comp_vals = true_comp_vals[::-1, ::-1]

    if which != "full":
        if group_size is None:
            group_size = len(circuit)
        custom_idx = custom_idx_groups_of_n(len(circuit), group_size)
    else:
        custom_idx = None
    sv_comp_vals = mme_func(
        circuit,
        comp_states,
        circuit.parameters,
        "statevector",
        n_shots,
        custom_idx=custom_idx,
        which_hadamard_test=which,
    )
    sv_trace_dist = trace_dist(true_comp_vals, sv_comp_vals)
    print(f"Statevector {sv_trace_dist:.5e}")

    print(np.round(true_comp_vals, 3))
    print(np.round(sv_comp_vals, 3))

    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, which=backend_name)
    fake_comp_vals = mme_func(
        circuit,
        comp_states,
        circuit.parameters,
        fake_backend,
        n_shots,
        custom_idx=custom_idx,
        which_hadamard_test=which,
    )
    print(np.round(fake_comp_vals, 3))
    fake_trace_dist = trace_dist(true_comp_vals, fake_comp_vals)
    print(f"Fake Backend ({fake_backend.name}) {fake_trace_dist:.5e}")

    real_comp_vals = None
    if run_on_hardware:
        real_backend = get_backend(circuit.num_qubits, backend_name)
        real_comp_vals = mme_func(
            circuit,
            comp_states,
            circuit.parameters,
            real_backend,
            n_shots,
            custom_idx=custom_idx,
            which_hadamard_test=which,
        )
        real_trace_dist = trace_dist(true_comp_vals, real_comp_vals)
        print(f"RealBackend ({real_backend.name}) {real_trace_dist:.5e}")

    var_dict = {
        "fake_backend": {
            "name": fake_backend.name,
            "trace_dist": fake_trace_dist,
            "values": fake_comp_vals,
        },
        "statevector": {
            "trace_dist": sv_trace_dist,
            "values": sv_comp_vals,
        },
        "system": {"n_qubits": circuit.num_qubits, "values": true_comp_vals},
    }
    if run_on_hardware:
        var_dict["real_backend"] = {
            "name": real_backend.name,
            "trace_dist": real_trace_dist,
            "values": real_backend,
        }

    return var_dict


def measure_sweep_1qp(n_shots=10000):
    n_qubits = 5
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

    d = measure_1qp_elements(circuit, n_shots=n_shots, group_size=1)
    return d


def measure_vqe_1qp(n_shots: int = 10000):
    dirpath = Path(HOME, "../../res_sim/1qp_small/vqe/")
    for dirname in os.listdir(dirpath):
        hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(
            Path(dirpath, dirname)
        )
        print(f"{hamiltonian.num_qubits} qubits, J/h {get_tfim_ratio(hamiltonian)}")
        circuit_with_params = apply_parameters(circuit, weights[-1])

        measure_1qp_elements(circuit_with_params, n_shots=n_shots, group_size=2)


if __name__ == "__main__":
    # 1e9 shots is too much for fake backend, causing:
    # ERROR: [Experiment 0] bad allocation [Experiment 1] bad allocation , ERROR: bad allocation' qiskit
    # full Hadamard test for 4 qubits at 1e7 shots is GARBAGE on IBM fakebackend
    # real backends should be worse

    dry_run = True
    edm = ExperimentDataManager("1qp_sector_vqe_sweep_hw", dry_run=dry_run)

    n_shots = int(1e2)
    print(f"n_shots: {n_shots}")
    print("=== Sweep ===")
    measure_sweep_1qp(n_shots)
    print("=== VQE ===")
    measure_vqe_1qp(n_shots)
