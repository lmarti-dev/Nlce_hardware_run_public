from pathlib import Path

import numpy as np

from qcomp_nlce.utils.folders import res_sim_dir
from qcomp_nlce.utils.results import show_diffs
from qcomp_nlce.utils.workflow_utils import (
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
from fauplotstyle import style


def workflow_nq_ibm(dry_run):

    J = 1
    which_backends = [
        # "real_backend",
        "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "IBM_GENERIC"
    n_shots = int(1e6)
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


def workflow_nq_qexa(dry_run):

    J = 1
    which_backends = [
        # "real_backend",
        "fake_backend",
        # "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "QEXA"
    n_shots = int(1e5)
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
        n_reps=1,
    )


def workflow_nq_iqm_manual(dry_run):
    J = 1
    which_backends = [
        # "real_backend",
        "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "IQM_PERFECT"
    n_shots = int(1e6)
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


def workflow_nq_aqt(dry_run):

    J = 0.8
    which_backends = [
        # "real_backend",
        "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "AQT"
    n_shots = int(2000)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["vqe", "sweep"],
        J,
        n_reps=1,
        n_steps=10,
    )
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["sweep"],
        J,
        n_reps=1,
        n_steps=20,
    )


def workflow_nq_aqt_ladder(dry_run):

    J = 0.20
    which_backends = [
        # "real_backend",
        # "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "AQT"
    n_shots = int(200)
    which_hadamard_test = "stefan"
    n_qubits_max = 4
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["vqe"],
        J,
        which_model="TFIM_ladder",
        n_reps=10,
    )


def workflow_nq_aqt_sweep(dry_run):

    J = 1
    which_backends = [
        "real_backend",
        "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "AQT_BETTER_100"
    n_shots = int(2000)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["sweep"],
        J,
        n_steps=10,
    )
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["sweep"],
        J,
        n_steps=20,
    )


def workflow_nq_aqt_hl(dry_run):

    J = 0.8
    which_backends = [
        "real_backend",
        "fake_backend",
        "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "AQT"
    n_shots = int(200)
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
        which_model="TFIM_hl_10",
        n_reps=10,
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
    which_model: str = "TFIM",
    n_reps: int = 1,
):
    # avoid the "unused arguments" warning
    # warnings.filterwarnings("ignore")
    ks = np.linspace(0, np.pi, 21)
    group_size = None
    ladder_model = "ladder" in which_model
    if ladder_model:
        qubits_max_str = f"{n_qubits_max}x2"
        qubits_min_str = f"{n_qubits_max - 1}x2"

    else:
        qubits_max_str = str(n_qubits_max)
        qubits_min_str = str(n_qubits_max - 1)

    vqe_l_dir = f"{qubits_max_str}q_{get_optimal_n_layers(n_qubits_max,ladder_model)}l"
    vqe_s_dir = (
        f"{qubits_min_str}q_{get_optimal_n_layers(n_qubits_max-1,ladder_model)}l"
    )

    model_dir = f"vqe_{int(J*100)}j"

    if "HL" in which_model:
        if len(which_model.split("_")) == 3:
            hl = int(which_model.split("_")[-1])
        else:
            hl = 10
    else:
        hl = 0
    model_dir += f"_{hl}hl"
    if "ladder" in which_model:
        model_dir += "_ladder"

    gs_dirname_l = Path(res_sim_dir(), f"gs_{model_dir}/vqe/{vqe_l_dir}")
    gs_dirname_s = Path(res_sim_dir(), f"gs_{model_dir}/vqe/{vqe_s_dir}")

    elem_dirname_l = Path(
        res_sim_dir(),
        f"elem_{model_dir}/vqe/cost_offdiag/{vqe_l_dir}",
    )
    elem_dirname_s = Path(
        res_sim_dir(),
        f"elem_{model_dir}/vqe/cost_offdiag/{vqe_s_dir}",
    )

    edm = ExperimentDataManager(
        f"vqe_workflow_{which_model}_hadam_{which_hadamard_test}_{provider_name}_{n_shots}shots",
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
            "which_model": which_model,
            "n_steps": n_steps,
            "n_mc_samples": n_mc_samples,
            "which_algorithms": which_algorithms,
            "which_backends": which_backends,
            "which_hadamard_test": which_hadamard_test,
            "n_reps": n_reps,
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
            which_model,
            n_reps,
        )

        out_dict["results"]["sweep"] = {
            "dict_l": sweep_dict_l,
            "dict_s": sweep_dict_s,
            "disp": sweep_disp,
        }
        mc_disp = mc_disp_from_res_dict(
            ks, out_dict, "sweep", n_mc_samples, to_eotm=True
        )
        out_dict["results"]["sweep"]["mc_disp"] = mc_disp

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
            n_reps=n_reps,
            which_model=which_model,
        )

        out_dict["results"]["vqe"] = {
            "dict_l": vqe_dict_l,
            "dict_s": vqe_dict_s,
            "disp": vqe_disp,
        }
        mc_disp = mc_disp_from_res_dict(ks, out_dict, "vqe", n_mc_samples, to_eotm=True)
        out_dict["results"]["vqe"]["mc_disp"] = mc_disp

    ed_disp = tfim_ed_dispersion(n_qubits_max, J, ks, hl=hl, ladder=ladder_model)
    out_dict["results"]["ed"] = {"disp": {"statevector": ed_disp}}
    edm.save_dict(out_dict)

    edm.var_dump(**out_dict["inputs"])

    fig = plot_all_backends_dispersions_montecarlo(out_dict)

    edm.save_figure(fig, fig_shape="double-y")
    # show_diffs(out_dict, "fake_backend")
    plt.show()


if __name__ == "__main__":
    style()
    workflow_nq_aqt(False)
