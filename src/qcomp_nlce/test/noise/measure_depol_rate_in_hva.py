from qcomp_nlce.utils import (
    get_hva_vqe,
    backend_sample,
    get_appropriate_fake_backend,
    TFIM,
    apply_parameters,
    get_counts_from_res,
    load_optimized_vqe,
    trace_out_estimation,
    res_sim_dir,
    PROVIDER_ALIAS,
    backend_estimate,
    res_to_lists,
    estimate_ham_with_reps,
)
import numpy as np
from qiskit_aqt_provider.primitives import AQTSampler
from pathlib import Path

from qiskit import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp


def compute_hva_circuit_depol_by_half(
    backend,
    hamiltonian: SparsePauliOp,
    n_layers: int,
    parameters: np.ndarray,
    provider_name: PROVIDER_ALIAS,
):
    depol_circ = get_hva_vqe(
        hamiltonian=hamiltonian, n_layers=n_layers // 2, unique_weights=True
    )
    n_qubits = hamiltonian.num_qubits

    ts_depol_circ = apply_parameters(
        depol_circ, parameters=parameters[: len(depol_circ.parameters)]
    )
    ts_depol_circ.append(ts_depol_circ.inverse(), range(n_qubits))
    ts_depol_circ.measure_all()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    isa_circuit = pm.run(ts_depol_circ)
    sampler = AQTSampler(backend)
    job = sampler.run([isa_circuit], parameter_values=None, shots=n_shots)
    res = job.result()

    counts = get_counts_from_res(res, n_qubits)

    z_vals = []
    z_stds = []

    for jj in range(hamiltonian.num_qubits):
        # set last arg to 1 to enable mitigation.
        val, std = trace_out_estimation(counts, [jj], -1)
        z_vals.append(val)
        z_stds.append(std)

    depol_rate = 1 - np.mean(z_vals)
    depol_rate_unc = np.std(z_vals)
    # there should be an std on the std, but I don't think it should make too much of a diff.

    print(f"Depol rate: {depol_rate:.4f}, std: {depol_rate_unc:.4f}")
    return depol_rate, depol_rate_unc


if __name__ == "__main__":
    J = 0.8
    hl = 0
    n_qubits = 5
    n_shots = 2000
    provider_name = "AQT"
    n_layers = 6

    vqe_path = Path(res_sim_dir(), "gs_vqe_80j_10hl/vqe/5q_6l")
    hamiltonian, gs_circuit, gs_weights, ground_energy = load_optimized_vqe(vqe_path)
    params = gs_weights[-1]
    backend = get_appropriate_fake_backend(n_qubits, provider_name)
    depol_rate, depol_rate_std = compute_hva_circuit_depol_by_half(
        backend, hamiltonian, n_layers, params, provider_name
    )
    gs_circuit = apply_parameters(gs_circuit, params)
    vals_fb, stds_fb = estimate_ham_with_reps(
        hamiltonian,
        gs_circuit,
        n_shots,
        backend,
        1,
        "fake_backend",
    )

    job_sv = backend_estimate(backend, hamiltonian, [gs_circuit], params, n_shots)
    res_sv = job_sv.result()

    vals_sv, stds_sv = res_to_lists(res_sv)

    print(
        f"sv: {vals_sv}, fb: {vals_fb}, depol: {depol_rate}, corrected: {vals_fb/(1-depol_rate)}"
    )
