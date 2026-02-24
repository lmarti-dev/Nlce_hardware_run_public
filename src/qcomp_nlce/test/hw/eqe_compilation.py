from qcomp_nlce.utils import (
    build_stefan_circuits_sample,
    load_optimized_vqe,
    res_sim_dir,
    model_str,
    get_optimal_n_layers,
    apply_parameters,
    get_backend,
    pending_jobs,
    hadam_diff,
    get_counts_from_res,
)
from qiskit import transpile

from pathlib import Path
from data_manager import ExperimentDataManager


edm = ExperimentDataManager("EQE_compilation_test", dry_run=False, project="nlce")

J = 1
hl = 0
n_qubits = 8
provider_name = "EQE"
n_shots = int(1e5)

which_model = model_str(J, hl, False)
model_dir = "vqe_" + which_model

d_out = {}

vqe_dir = f"{str(n_qubits)}q_{get_optimal_n_layers(n_qubits,False)}l"
gs_dirname = Path(res_sim_dir(), f"gs_{model_dir}/vqe/{vqe_dir}")

hamiltonian, vqe_circuit, weights, ground_energy = load_optimized_vqe(gs_dirname)
vqe_circuit = apply_parameters(vqe_circuit, weights[-1])


stefan_test_circuits = build_stefan_circuits_sample(
    vqe_circuit, None, "00001000", "00100000"
)

backend = get_backend(n_qubits, provider_name)

pending = pending_jobs(n_qubits, provider_name)
if pending > 0:
    raise ValueError(f"There are {pending} pending jobs, wait!")


# try this
# try:
#     job = backend.run(circs, shots=n_shots, no_modify=False)
#     res = job.result()
#     keys = {"p0": "0" * (n_qubits + 1), "p1": "0" * (n_qubits) + "1"}
#     res = job.result()
#     re, rstd = hadam_diff(get_counts_from_res(res, n_qubits, 0), n_shots, keys=keys)
#     im, imstd = hadam_diff(get_counts_from_res(res, n_qubits, 1), n_shots, keys=keys)
#     steval, stestd = re + 1j * im, rstd + 1j * imstd
# except Exception as e:
#     edm.save_dict({"Exception": e.__str__(), "method": "cloud_compile"})

# try thisthat
try:
    initial_layout = list(range(n_qubits + 1))
    isa_circuit = transpile(
        stefan_test_circuits,
        backend=backend,
        initial_layout=initial_layout,
        optimization_level=3,
    )
    job = backend.run(isa_circuit, shots=n_shots, no_modify=True)
    res = job.result()
    keys = {"p0": "0" * (n_qubits + 1), "p1": "0" * (n_qubits) + "1"}
    re, rstd = hadam_diff(get_counts_from_res(res, n_qubits, 0), n_shots, keys=keys)
    im, imstd = hadam_diff(get_counts_from_res(res, n_qubits, 1), n_shots, keys=keys)
    steval, stestd = re + 1j * im, rstd + 1j * imstd
except Exception as e:
    edm.save_dict({"Exception": e.__str__(), "method": "local_compile"})
