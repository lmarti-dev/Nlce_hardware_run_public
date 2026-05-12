from src.qcomp_nlce.utils import (
    TFIM,
    get_min_gap,
    get_tfim_sweep_circuit,
    sweep_time_from_steps,
    circuit_matrix_elements,
    hamming_weight_states,
    load_optimized_vqe,
    apply_parameters,
    res_sim_dir,
    circuit_to_matrix,
    get_1qp_mat,
)

from fauplotstyle import style
from src.qcomp_nlce.utils.results import pretty_print
from pathlib import Path
import numpy as np

from data_manager import ExperimentDataManager

import matplotlib.pyplot as plt


edm = ExperimentDataManager("circuit_symmetries_nlce", project="nlce", dry_run=False)


n_qubits = 5
J = 0.8
hl = 0
ladder = False


states = hamming_weight_states(n_qubits, 1)


ham_j, ham_h = TFIM(
    n_qubits, 1, J, split_parts=True, longitudinal_field=hl, ladder=ladder
)


hamiltonian = ham_j + ham_h

vals, vecs = np.linalg.eigh(hamiltonian.to_matrix())

submat = get_1qp_mat(vecs, False)

pretty_print(np.abs(submat - submat.T))


min_gap = get_min_gap(hamiltonian)

diffs = []
steps = np.arange(2, 501, 5)

print("SWEEP")
for n_steps in steps:
    total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )
    mat = circuit_matrix_elements(circuit, states)
    print(f"n_steps: {n_steps}")
    diff = np.abs(mat - mat.T)
    pretty_print(diff)
    diffs.append(np.sum(diff))


style()
fig, ax = plt.subplots()
ax.plot(steps, diffs)
ax.set_xscale("log")
ax.set_xlabel("Number of steps")
ax.set_ylabel(r"$|\tilde{O}-\tilde{O}^T|$")
edm.save_figure(fig, filename="asp_symmetry_error")

print("VQE")
vqe_dirname = "elem_vqe_80j_0hl/vqe/cost_offdiag/5q_3l"
elem_dirname_l = Path(res_sim_dir(), vqe_dirname)
hamiltonian1, elem_circuit, elem_weights, ground_energy = load_optimized_vqe(
    elem_dirname_l
)
elem_circuit = apply_parameters(elem_circuit, elem_weights[-1])
mat = circuit_matrix_elements(elem_circuit, states)
diff = np.abs(mat - mat.T)
print(np.sum(diff))
edm.var_dump(n_qubits=n_qubits, J=J, hl=hl, ladder=ladder, vqe_dirname=vqe_dirname)

edm.save_dict({"steps": steps, "diffs": diffs, "vqe_diff": np.sum(diff)})

plt.show()
