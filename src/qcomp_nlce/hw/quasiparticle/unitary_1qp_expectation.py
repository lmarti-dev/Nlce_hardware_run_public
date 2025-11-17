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
)
from pathlib import Path

HOME = Path(__file__).parent


#  this needs

#


def measure_1qp_expectations(
    circuit: QuantumCircuit,
    backend_name: str = "IBM",
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
) -> np.ndarray:

    mme_func = measure_expectations
    which = "off-diag"

    n_qubits = circuit.num_qubits
    print(f"Minimal trace error (only shot noise) {.5*(n_qubits)/np.sqrt(n_shots):.4e}")
    comp_states = hamming_weight_states(n_qubits, 1)
    true_comp_vals = circuit_matrix_elements(circuit, comp_states)

    # the matrix you get otherwise is indexed backwards wrt to what you get
    # from the circuit-to-matrix qiskit function
    true_comp_vals = true_comp_vals[::-1, ::-1]

    custom_idx = custom_idx_groups_of_n(len(circuit), 2)
    sv_comp_vals = mme_func(
        circuit,
        comp_states,
        circuit.parameters,
        "statevector",
        n_shots,
        custom_idx=custom_idx,
        which=which,
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
        which=which,
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
            custom_idx=None,
            which=which,
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
