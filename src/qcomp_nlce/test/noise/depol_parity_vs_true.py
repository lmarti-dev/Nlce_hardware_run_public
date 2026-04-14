from src.qcomp_nlce.utils.sweep import sweep_time_from_steps, get_tfim_sweep_circuit
from src.qcomp_nlce.utils.hamiltonian import TFIM

from src.qcomp_nlce.utils import (
    OFFDIAG_SYMS,
    estimate_ham_with_reps,
    index_bits,
    prep_circ_superpos,
    compute_off_diagonal_element,
    hamming_weight_states,
    reorder_matrix_elems,
    get_appropriate_fake_backend,
    circ_ham_expectation_values,
    apply_parameters,
    load_optimized_vqe,
    model_str,
    res_sim_dir,
    get_optimal_n_layers,
    baptize_circuit,
    circuit_to_matrix,
)

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from pathlib import Path


from data_manager import ExperimentDataManager


def circ_ham_expect(circ: QuantumCircuit, hmat: np.ndarray):
    cmat = circuit_to_matrix(circ)

    sv = cmat[:, 0]
    return sv.conj().dot(hmat @ sv)


if __name__ == "__main__":
    dry_run = False
    edm = ExperimentDataManager("depol_parity_vs_true", dry_run=dry_run, project="nlce")

    n_qubits = 5
    J = 0.8
    hl = 0
    ladder = False
    n_shots = 10
    n_reps = 2
    mitigate_depolarization = False
    which_algorithm = "estimator"
    provider_name = "AQT"
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
        "vqe",
        n_qubits,
        provider_name,
        n_shots,
        "none",
    )
    circuit.metadata["tmpname"] = name_part

    diag_vals = np.zeros((n_qubits,))
    diag_stds = np.zeros((n_qubits,))
    diag_true_vals = np.zeros((n_qubits,))
    diag_parity_vals = np.zeros((n_qubits,))
    diag_parity_true_vals = np.zeros((n_qubits,))
    diag_parity_stds = np.zeros((n_qubits,))
    offd_vals = {k: np.zeros((n_qubits, n_qubits)) for k in OFFDIAG_SYMS}
    offd_true_vals = {k: np.zeros((n_qubits, n_qubits)) for k in OFFDIAG_SYMS}
    offd_stds = {k: np.zeros((n_qubits, n_qubits)) for k in OFFDIAG_SYMS}
    offd_parity_vals = {k: np.zeros((n_qubits, n_qubits)) for k in OFFDIAG_SYMS}
    offd_parity_true_vals = {k: np.zeros((n_qubits, n_qubits)) for k in OFFDIAG_SYMS}
    offd_parity_stds = {k: np.zeros((n_qubits, n_qubits)) for k in OFFDIAG_SYMS}

    states = hamming_weight_states(n_qubits, 1)
    parity_op = SparsePauliOp("Z" * n_qubits)

    hmat = hamiltonian.to_matrix()
    parity_op_mat = parity_op.to_matrix()

    for ind_x, x in enumerate(states):
        for ind_y, y in enumerate(states[ind_x:]):
            ind_y = ind_x + ind_y

            if ind_x == ind_y:
                circ_out = QuantumCircuit(circuit.num_qubits)
                x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
                for x_ind in x_inds:
                    circ_out.x(x_ind)
                circ_out.compose(circuit, inplace=True)
                circ_out.metadata["tmpname"] = circuit.metadata["tmpname"]
                val, std = estimate_ham_with_reps(
                    hamiltonian,
                    circ_out,
                    n_shots,
                    backend,
                    which_backend=backend,
                    n_reps=n_reps,
                    which_algorithm=which_algorithm,
                    mitigate_depolarization=mitigate_depolarization,
                )

                parity_val, parity_std = estimate_ham_with_reps(
                    parity_op,
                    circ_out,
                    n_shots,
                    backend,
                    which_backend=backend,
                    n_reps=n_shots,
                    which_algorithm=which_algorithm,
                    mitigate_depolarization=mitigate_depolarization,
                )

                print(f"{ind_x} - {ind_y} - {complex(val):.4f} ({complex(std):.4f})")

                diag_vals[ind_x] = val
                diag_true_vals[ind_x] = circ_ham_expect(circ_out, hmat)
                diag_stds[ind_x] = std
                diag_parity_vals[ind_x] = parity_val
                diag_parity_true_vals[ind_x] = circ_ham_expect(circ_out, parity_op_mat)
                diag_parity_stds[ind_x] = parity_std

            else:
                for k in OFFDIAG_SYMS:
                    sind_x = str(ind_x) + k
                    sind_y = str(ind_y) + k

                    circ_p = prep_circ_superpos(x, y, k)
                    circ_p.compose(circuit, inplace=True)
                    circ_p.metadata["tmpname"] = circuit.metadata["tmpname"]
                    t_val, t_std = estimate_ham_with_reps(
                        hamiltonian,
                        circ_p,
                        n_shots,
                        backend,
                        which_backend=backend,
                        n_reps=n_reps,
                        which_algorithm=which_algorithm,
                        mitigate_depolarization=mitigate_depolarization,
                    )

                    parity_val, parity_std = estimate_ham_with_reps(
                        parity_op,
                        circ_p,
                        n_shots,
                        backend,
                        which_backend=backend,
                        n_reps=n_reps,
                        which_algorithm=which_algorithm,
                        mitigate_depolarization=mitigate_depolarization,
                    )

                    true_val = circ_ham_expect(circ_p, hmat)
                    parity_true_val = circ_ham_expect(circ_p, parity_op_mat)

                    offd_vals[k][ind_x, ind_y] = t_val
                    offd_vals[k][ind_y, ind_x] = t_val.conj()
                    offd_true_vals[k][ind_x, ind_y] = true_val
                    offd_true_vals[k][ind_y, ind_x] = true_val.conj()
                    offd_stds[k][ind_x, ind_y] = t_std
                    offd_stds[k][ind_y, ind_x] = t_std.conj()
                    offd_parity_vals[k][ind_x, ind_y] = parity_val
                    offd_parity_vals[k][ind_y, ind_x] = parity_val.conj()
                    offd_parity_true_vals[k][ind_x, ind_y] = parity_true_val
                    offd_parity_true_vals[k][ind_y, ind_x] = parity_true_val.conj()
                    offd_parity_stds[k][ind_x, ind_y] = parity_std
                    offd_parity_stds[k][ind_y, ind_x] = parity_std.conj()

    edm.save_dict(
        {
            "inputs": {
                "J": J,
                "hl": hl,
                "ladder": ladder,
                "n_qubits": n_qubits,
                "n_shots": n_shots,
                "n_reps": n_reps,
                "which_algorithm": which_algorithm,
                "mitigate_depolarization": mitigate_depolarization,
                "provider_name": provider_name,
            },
            "results": {
                "diag_vals": diag_vals,
                "diag_true_vals": diag_true_vals,
                "diag_stds": diag_stds,
                "diag_parity_vals": diag_parity_vals,
                "diag_parity_true_vals": diag_parity_true_vals,
                "diag_parity_stds": diag_parity_stds,
                "offd_vals": offd_vals,
                "offd_true_vals": offd_true_vals,
                "offd_stds": offd_stds,
                "offd_parity_vals": offd_parity_vals,
                "offd_parity_true_vals": offd_parity_true_vals,
                "offd_parity_stds": offd_parity_stds,
            },
        }
    )
