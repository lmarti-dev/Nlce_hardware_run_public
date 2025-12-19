from pathlib import Path

import numpy as np

from qcomp_nlce.utils.folders import res_sim_dir
from qcomp_nlce.utils.workflow_1d import (
    vqe_dispersion,
    tfim_ed_dispersion,
    sweep_dispersion,
    plot_all_backends_dispersions_montecarlo,
)
from qcomp_nlce.utils.vqe import get_optimal_n_layers
from qcomp_nlce.utils.stats import mc_disp_from_res_dict
from qcomp_nlce.utils.hardware import PROVIDER_ALIAS, BACKEND_TYPES
from qcomp_nlce.utils.overlap import HADAMARD_TESTS
from qcomp_nlce.utils.constants import ALGORITHMS_ALIAS
from data_manager import ExperimentDataManager

import matplotlib.pyplot as plt


def workflow_nq_qexa(dry_run):

    J = 1
    which_backends = [
        "real_backend",
        "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "QEXA"
    n_shots = int(2e5)
    which_hadamard_test = "stefan_samples"
    n_qubits_max = 5
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["vqe"],
        J,
    )


def workflow_nq_iqm(dry_run):

    J = 1
    which_backends = [
        # "real_backend",
        "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "IQM"
    n_shots = int(2e5)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["vqe"],
        J,
    )


def workflow_nq_aqt(dry_run):

    J = 1
    which_backends = [
        "fake_backend",
        "real_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "AQT"
    n_shots = int(200)
    which_hadamard_test = "estimator"
    n_qubits_max = 5
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["vqe"],
        J,
    )


def workflow_nq(
    dry_run: bool,
    n_qubits_max: int,
    which_hadamard_test: HADAMARD_TESTS,
    provider_name: PROVIDER_ALIAS,
    n_shots: int,
    which_backends: BACKEND_TYPES,
    which_algorithms: list[ALGORITHMS_ALIAS],
    J: float,
    n_steps: int = 1,
    n_mc_samples: int = 10000,
):
    # avoid the "unused arguments" warning
    # warnings.filterwarnings("ignore")
    ks = np.linspace(0, np.pi, 21)
    group_size = None

    vqe_l_dir = f"{n_qubits_max}q_{get_optimal_n_layers(n_qubits_max)}l"
    vqe_s_dir = f"{n_qubits_max-1}q_{get_optimal_n_layers(n_qubits_max-1)}l"

    gs_dirname_l = Path(res_sim_dir(), f"gs_vqe_{int(J*100)}j/vqe/{vqe_l_dir}")
    gs_dirname_s = Path(res_sim_dir(), f"gs_vqe_{int(J*100)}j/vqe/{vqe_s_dir}")

    elem_dirname_l = Path(
        res_sim_dir(),
        f"elem_vqe_{int(J*100)}j/vqe/cost_offdiag/{vqe_l_dir}",
    )
    elem_dirname_s = Path(
        res_sim_dir(),
        f"elem_vqe_{int(J*100)}j/vqe/cost_offdiag/{vqe_s_dir}",
    )

    edm = ExperimentDataManager(
        f"vqe_workflow_tfim_hadam_{which_hadamard_test}_{provider_name}_{n_shots}shots",
        dry_run=dry_run,
    )

    out_dict = {
        "results": {},
        "inputs": {
            "ks": ks,
            "J": J,
            "provider_name": provider_name,
            "group_size": group_size,
            "n_qubits": n_qubits_max,
            "n_shots": n_shots,
        },
    }
    if "sweep" in which_algorithms:
        print("== Sweep ==")
        sweep_dict_l, sweep_dict_s, sweep_disp = sweep_dispersion(
            n_qubits_max,
            J,
            n_steps,
            which_backends,
            provider_name,
            n_shots,
            ks,
            group_size,
            which_hadamard_test,
        )

        out_dict["results"]["sweep"] = {
            "dict_l": sweep_dict_l,
            "dict_s": sweep_dict_s,
            "disp": sweep_disp,
        }

    if "vqe" in which_algorithms:
        print("== VQE ==")
        vqe_dict_l, vqe_dict_s, vqe_disp = vqe_dispersion(
            gs_dirname_l,
            elem_dirname_l,
            gs_dirname_s,
            elem_dirname_s,
            which_backends,
            provider_name,
            n_shots,
            ks,
            group_size,
            which_hadamard_test,
        )

        out_dict["results"]["vqe"] = {
            "dict_l": vqe_dict_l,
            "dict_s": vqe_dict_s,
            "disp": vqe_disp,
        }
        mc_disp = mc_disp_from_res_dict(ks, out_dict, "vqe", n_mc_samples, to_eotm=True)
        out_dict["results"]["vqe"]["mc_disp"] = mc_disp

    ed_disp = tfim_ed_dispersion(n_qubits_max, J, ks)
    out_dict["results"]["ed"] = {"disp": {"statevector": ed_disp}}
    edm.save_dict(out_dict)

    fig = plot_all_backends_dispersions_montecarlo(out_dict)

    edm.save_figure(fig, fig_shape="double-y")
    plt.show()


if __name__ == "__main__":
    workflow_nq_qexa(True)
