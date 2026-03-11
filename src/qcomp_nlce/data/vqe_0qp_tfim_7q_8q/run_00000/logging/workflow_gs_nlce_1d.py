from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    loop_measure_ground_state_energy_subspace_sampling_v2,
)
from qcomp_nlce.utils import (
    load_optimized_vqe,
    res_sim_dir,
    model_str,
    get_optimal_n_layers,
    apply_parameters,
    BACKEND_TYPES,
    baptize_circuit,
)
from pathlib import Path
from data_manager import ExperimentDataManager
import matplotlib.pyplot as plt


from fauplotstyle import style

from qcomp_nlce.utils.workflow_utils import plot_0qp_gs_dispersion


def gs_subspace_run(
    n_qubits: int,
    J: float,
    hl: float,
    n_shots: int,
    which_backends: BACKEND_TYPES,
):

    which_model = model_str(J, hl, False)
    model_dir = "vqe_" + which_model

    vqe_dir = f"{str(n_qubits)}q_{get_optimal_n_layers(n_qubits,False)}l"
    gs_dirname = Path(res_sim_dir(), f"gs_{model_dir}/vqe/{vqe_dir}")

    hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(gs_dirname)
    circuit = apply_parameters(circuit, weights[-1])
    name_part = baptize_circuit(
        f"gs_dim1d_{which_model}",
        "vqe",
        n_qubits,
        "EQE",
        n_shots,
        "nohadam",
    )
    circuit.metadata["tmpname"] = name_part

    res_dict = loop_measure_ground_state_energy_subspace_sampling_v2(
        hamiltonian,
        circuit,
        which_backends,
        n_shots=n_shots,
        provider_name="EQE",
        solve=True,
    )

    return res_dict


def workflow_eqe(dry_run):
    n_shots = int(1e4)
    which_backends: BACKEND_TYPES = [
        "statevector",
        "fake_backend",
        "real_backend",
    ]
    edm = ExperimentDataManager(
        f"nlce_gs_1d_tfim_0qp_{n_shots}shots",
        project="nlce",
        dry_run=dry_run,
    )

    j_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    qubit_sizes = (7, 8)
    d = {
        "results": {
            be: {j: {v: 1000 for v in qubit_sizes} for j in j_range}
            for be in [*which_backends, "system"]
        }
    }
    d["inputs"] = {"j_range": j_range, "qubit_sizes": qubit_sizes}

    for n_qubits in qubit_sizes:
        for J in j_range:
            res_dict = gs_subspace_run(
                n_qubits,
                J,
                hl=0,
                n_shots=n_shots,
                which_backends=which_backends,
            )
            for be in res_dict:
                d["results"][be][J][n_qubits] = res_dict[be]

    fig = plot_0qp_gs_dispersion(d)

    edm.save_dict(d)

    edm.save_figure(fig, "1d_tfim_chain_0qp_gs", add_timestamp=False)

    plt.show()


def test():
    n_shots = int(1e4)
    which_backends: BACKEND_TYPES = ["statevector", "fake_backend", "real_backend"]
    n_qubits = 8
    J = 1.0
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
    style()
    workflow_eqe(False)
