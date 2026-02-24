from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    measure_ground_state_energy_subspace_sampling_v2,
)
from qcomp_nlce.utils import (
    load_optimized_vqe,
    res_sim_dir,
    model_str,
    get_optimal_n_layers,
    apply_parameters,
    BACKEND_TYPES,
)
from pathlib import Path
from data_manager import ExperimentDataManager


def gs_subspace_run(
    n_qubits: int,
    J: float,
    hl: float,
    n_shots: int,
    which_backends: BACKEND_TYPES,
    dry_run: bool,
):
    edm = ExperimentDataManager(
        f"nlce_gs_1d_tfim_{n_qubits}q_{str(J*100)}j_{str(100*hl)}hl_{n_shots}shots",
        project="nlce",
        dry_run=dry_run,
    )

    which_model = model_str(J, hl, False)
    model_dir = "vqe_" + which_model

    d_out = {}

    vqe_dir = f"{str(n_qubits)}q_{get_optimal_n_layers(n_qubits,False)}l"
    gs_dirname = Path(res_sim_dir(), f"gs_{model_dir}/vqe/{vqe_dir}")

    hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(gs_dirname)
    circuit = apply_parameters(circuit, weights[-1])

    for which_backend in which_backends:
        res_dict = measure_ground_state_energy_subspace_sampling_v2(
            hamiltonian,
            ground_energy,
            circuit,
            which_backend=which_backend,
            n_shots=n_shots,
            provider_name="EQE",
        )
        res_dict[which_backend] = d_out[which_backend]

    edm.save_dict(res_dict)

    return res_dict


def workflow_eqe():
    n_shots = int(1e4)
    which_backends: BACKEND_TYPES = ["statevector", "fake_backend", "real_backend"]
    j_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    dry_run = False
    for n_qubits in (7, 8):
        for J in j_range:
            gs_subspace_run(
                n_qubits,
                J,
                hl=0,
                n_shots=n_shots,
                which_backends=which_backends,
                dry_run=dry_run,
            )


def test():
    n_shots = int(1e4)
    which_backends: BACKEND_TYPES = ["statevector", "fake_backend", "real_backend"]
    n_qubits = 8
    J=1.0
    dry_run = True
    gs_subspace_run(
                n_qubits,
                J,
                hl=0,
                n_shots=n_shots,
                which_backends=which_backends,
                dry_run=dry_run,
            )

if __name__ == "__main__":

