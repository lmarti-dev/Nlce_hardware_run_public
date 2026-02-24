from qcomp_nlce.utils import (
    build_stefan_circuits_sample,
    load_optimized_vqe,
    res_sim_dir,
    model_str,
    get_optimal_n_layers,
    apply_parameters,
    get_backend,
    pending_jobs,
)

from pathlib import Path

J = 1
hl = 0
n_qubits = 8

which_model = model_str(J, hl, False)
model_dir = "vqe_" + which_model

d_out = {}

vqe_dir = f"{str(n_qubits)}q_{get_optimal_n_layers(n_qubits,False)}l"
gs_dirname = Path(res_sim_dir(), f"gs_{model_dir}/vqe/{vqe_dir}")

hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(gs_dirname)
circuit = apply_parameters(circuit, weights[-1])


circs = build_stefan_circuits_sample(circuit, None, "00001000", "00100000")
