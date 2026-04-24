from src.qcomp_nlce.utils.sweep import sweep_time_from_steps, get_tfim_sweep_circuit
from src.qcomp_nlce.utils.hamiltonian import TFIM

from src.qcomp_nlce.utils import (
    OFFDIAG_SYMS,
    estimate_ham_with_reps,
    index_bits,
    prep_circ_superpos,
    measure_matrix_elements,
    hamming_weight_states,
    reorder_matrix_elems,
    get_appropriate_fake_backend,
    apply_parameters,
    load_optimized_vqe,
    model_str,
    res_sim_dir,
    get_optimal_n_layers,
    baptize_circuit,
    circuit_to_matrix,
    circuit_matrix_elements,
)

import numpy as np
from qiskit import QuantumCircuit

from pathlib import Path


from data_manager import ExperimentDataManager


def circ_ham_expect(circ: QuantumCircuit, hmat: np.ndarray):
    cmat = circuit_to_matrix(circ)

    sv = cmat[:, 0]
    return sv.conj().dot(hmat @ sv)


if __name__ == "__main__":
    dry_run = False
    edm = ExperimentDataManager("depol_omatrix", dry_run=dry_run, project="nlce")

    n_qubits = 5
    J = 0.8
    hl = 0
    ladder = False
    n_shots = 2000
    n_reps = 20
    mitigate_depolarization = False
    provider_name = "AQT"
    which_backend = "fake_backend"
    backend = get_appropriate_fake_backend(n_qubits, provider_name)

    algorithm = "vqe"

    if algorithm == "vqe":
        which_model = model_str(J, hl, ladder)
        vqe_l_dir = f"{str(n_qubits)}q_{get_optimal_n_layers(n_qubits, ladder, hl)}l"
        model_dir = "vqe_" + which_model
        elem_dirname = elem_dirname_l = Path(
            res_sim_dir(),
            f"elem_{model_dir}/vqe/cost_offdiag/{vqe_l_dir}",
        )
        hamiltonian, elem_circuit, elem_weights, ground_energy = load_optimized_vqe(
            elem_dirname
        )
        n_qubits = hamiltonian.num_qubits
        circuit = apply_parameters(elem_circuit, elem_weights[-1])

    elif algorithm == "sweep":
        ham_j, ham_h = TFIM(
            n_qubits=n_qubits,
            h=1,
            J=J,
            longitudinal_field=hl,
            ladder=ladder,
            split_parts=True,
        )
        n_steps = 10
        hamiltonian = ham_h + ham_j
        total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
        circuit = get_tfim_sweep_circuit(
            ham_h,
            ham_j + ham_h,
            total_time=total_time,
            n_steps=n_steps,
        )

    name_part = baptize_circuit(
        f"dim1d_{which_model}",
        algorithm,
        n_qubits,
        provider_name,
        n_shots,
        "none",
    )
    circuit.metadata["tmpname"] = name_part

    states = hamming_weight_states(n_qubits, 1)

    which_hadamard_test = "stefan"

    true_mat = circuit_matrix_elements(circuit, states)
    oUo = circuit_to_matrix(circuit)[0, 0]
    true_mat *= oUo

    vals, stds = measure_matrix_elements(
        circuit=circuit,
        comp_states=states,
        parameters=None,
        backend=backend,
        which_backend=which_backend,
        n_shots=n_shots,
        which_hadamard_test=which_hadamard_test,
        n_reps=n_reps,
    )

    edm.save_dict(
        {
            "inputs": {
                "J": J,
                "hl": hl,
                "ladder": ladder,
                "n_qubits": n_qubits,
                "n_shots": n_shots,
                "n_reps": n_reps,
                "which_hadamard_test": which_hadamard_test,
                "mitigate_depolarization": mitigate_depolarization,
                "provider_name": provider_name,
            },
            "results": {"vals": vals, "stds": stds, "true_vals": true_mat},
        }
    )
