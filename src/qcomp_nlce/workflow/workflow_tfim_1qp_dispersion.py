from pathlib import Path

import numpy as np

from src.qcomp_nlce.utils.folders import res_sim_dir
from src.qcomp_nlce.utils.results import show_diffs
from src.qcomp_nlce.utils.workflow_utils import (
    vqe_dispersion,
    tfim_ed_dispersion,
    sweep_dispersion,
    plot_all_backends_dispersions_montecarlo,
)
from src.qcomp_nlce.utils.vqe import get_optimal_n_layers
from src.qcomp_nlce.utils.stats import mc_disp_from_res_dict
from src.qcomp_nlce.utils.hardware import (
    PROVIDER_ALIAS,
    BACKEND_TYPES,
    RUNS_POSTFIX,
)
from src.qcomp_nlce.utils.overlap import HADAMARD_TESTS, HAM_EXPECT_METHODS
from src.qcomp_nlce.utils.constants import ALGORITHMS_ALIAS
from src.qcomp_nlce.utils.hamiltonian import model_str
from data_manager import ExperimentDataManager

import matplotlib.pyplot as plt
from fauplotstyle import style


def workflow_nq_ibm(dry_run):
    J = 1
    which_backends = [
        # "real_backend",
        "fake_backend",
        # "statevector",
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
        which_algorithm_ham="sampling",
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


def workflow_nq_eqe(dry_run):
    J = 0.8
    which_backends = [
        "real_backend",
        # "fake_backend",
        # "statevector",
    ]
    provider_name: PROVIDER_ALIAS = "EQE"
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
        n_reps=1,
    )


def workflow_nq_aqt_100j(dry_run, which_backends: BACKEND_TYPES, run_postfix):
    return workflow_nq_aqt(dry_run, 1.0, which_backends, run_postfix)


def workflow_nq_aqt_50j(dry_run, which_backends: BACKEND_TYPES, run_postfix):
    return workflow_nq_aqt(dry_run, 0.5, which_backends, run_postfix)


def workflow_nq_aqt_30j(dry_run, which_backends: BACKEND_TYPES, run_postfix):
    return workflow_nq_aqt(dry_run, 0.3, which_backends, run_postfix)


def workflow_nq_aqt_80j(dry_run, which_backends: BACKEND_TYPES, run_postfix):
    return workflow_nq_aqt(dry_run, 0.8, which_backends, run_postfix)


def workflow_nq_aqt(dry_run, J, which_backends: BACKEND_TYPES, run_postfix):
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
        J=J,
        n_reps=10,
        n_steps=10,
        hl=0.0,
        postfix=run_postfix,
    )


def aer_depol(dry_run: bool):
    provider_name: PROVIDER_ALIAS
    n_shots = int(-1)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    which_backends = ["fake_backend", "statevector"]

    n_reps = 1

    for vals in (1, 2, 5, 10, 20, 50, 100, 1000, 2000):
        provider_name = f"AER_{vals}_{vals}"

        workflow_nq(
            dry_run,
            n_qubits_max,
            which_hadamard_test,
            provider_name,
            n_shots,
            which_backends,
            ["sweep"],
            J=0.8,
            n_reps=n_reps,
            n_steps=10,
            hl=0.0,
        )

        workflow_nq(
            dry_run,
            n_qubits_max,
            which_hadamard_test,
            provider_name,
            n_shots,
            which_backends,
            ["vqe"],
            J=0.8,
            n_reps=n_reps,
            n_steps=10,
            hl=0.0,
        )

        workflow_nq(
            dry_run,
            n_qubits_max,
            which_hadamard_test,
            provider_name,
            n_shots,
            which_backends,
            ["vqe"],
            J=1.0,
            n_reps=n_reps,
            n_steps=10,
            hl=0.0,
        )

        workflow_nq(
            dry_run,
            n_qubits_max,
            which_hadamard_test,
            provider_name,
            n_shots,
            which_backends,
            ["vqe"],
            J=0.5,
            n_reps=n_reps,
            n_steps=10,
            hl=0.1,
        )


def depol_errmit_vqe(dry_run):
    which_backends = ["fake_backend", "statevector"]
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
        ["vqe"],
        J=0.8,
        n_reps=1,
        n_steps=1,
        hl=0.0,
        which_algorithm_ham="estimator",
        mitigate_depolarization=False,
    )


def depol_vs_shots(dry_run):
    J = 0.8
    which_backends: BACKEND_TYPES = ["fake_backend", "statevector"]
    n_shots = int(2000)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    for provider_name in ("AQT_WORSE_2", "AQT", "AQT_BETTER_10", "AQT_BETTER_100"):
        for n_reps in (20,):
            workflow_nq(
                dry_run,
                n_qubits_max,
                which_hadamard_test,
                provider_name,
                n_shots,
                which_backends,
                ["vqe"],
                J=J,
                n_reps=n_reps,
                n_steps=10,
                hl=0.0,
            )


def n_reps_effect(dry_run, J, which_backends: BACKEND_TYPES):
    provider_name: PROVIDER_ALIAS = "AQT"
    n_shots = int(200)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    for n_reps in (2, 10, 40):
        workflow_nq(
            dry_run,
            n_qubits_max,
            which_hadamard_test,
            provider_name,
            n_shots,
            which_backends,
            ["vqe"],
            J=J,
            n_reps=n_reps,
            n_steps=10,
            hl=0.0,
        )


def workflow_nq_aqt_sweep_80j(dry_run, which_backends: BACKEND_TYPES):
    workflow_nq_aqt_sweep(0.8, dry_run, which_backends)


def workflow_nq_aqt_sweep_30j(dry_run, which_backends: BACKEND_TYPES):
    workflow_nq_aqt_sweep(0.3, dry_run, which_backends)


def workflow_nq_aqt_sweep(J, dry_run, which_backends: BACKEND_TYPES):
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
        ["sweep"],
        J=J,
        n_reps=10,
        n_steps=10,
        hl=0.0,
        mitigate_depolarization=False,
        which_algorithm_ham="sampling",
    )


def workflow_nq_aqt_sweepsims(dry_run: bool, J: float, which_backends: BACKEND_TYPES):
    n_shots = int(200)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    for n_steps in (10, 20):
        for provider_name in ("AQT", "AQT_BETTER_10", "AQT_BETTER_100"):
            workflow_nq(
                dry_run,
                n_qubits_max,
                which_hadamard_test,
                provider_name,
                n_shots,
                which_backends,
                ["sweep"],
                J=J,
                n_reps=10,
                n_steps=n_steps,
                hl=0.0,
            )


def sweep_convergence(dry_run: bool, J: float, which_backends: BACKEND_TYPES):
    n_shots = int(200)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    (2, 5, 10, 15, 20, 40)
    for n_steps in (3, 4, 7, 30, 50, 100):
        workflow_nq(
            dry_run,
            n_qubits_max,
            which_hadamard_test,
            "AQT",
            n_shots,
            which_backends,
            ["sweep"],
            J=J,
            n_reps=10,
            n_steps=n_steps,
            hl=0.0,
        )


def workflow_nq_aqt_ladder(dry_run, which_backends):
    J = 0.20

    provider_name: PROVIDER_ALIAS = "AQT"
    n_shots = int(200)
    which_hadamard_test = "stefan"
    n_qubits_max = 4
    ks = np.linspace(-np.pi, np.pi, 21)
    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["vqe"],
        J,
        n_reps=10,
        hl=0,
        ladder=True,
        ks=ks,
    )


def workflow_nq_aqt_hl(
    dry_run, which_backends: BACKEND_TYPES, run_postfix: RUNS_POSTFIX
):
    J = 0.5
    provider_name: PROVIDER_ALIAS = "AQT"
    n_shots = int(200)
    which_hadamard_test = "stefan"
    n_qubits_max = 5
    n_reps = 10
    hl = 0.1

    workflow_nq(
        dry_run,
        n_qubits_max,
        which_hadamard_test,
        provider_name,
        n_shots,
        which_backends,
        ["vqe"],
        J,
        n_reps=n_reps,
        n_steps=0,
        hl=hl,
        postfix=run_postfix,
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
    n_reps: int = 1,
    hl: float = 0,
    ladder: bool = False,
    ks: np.ndarray = None,
    symmetries: bool = True,
    postfix: RUNS_POSTFIX = "",
    which_algorithm_ham: HAM_EXPECT_METHODS = "estimator",
    mitigate_depolarization: bool = False,
):
    # avoid the "unused arguments" warning
    # warnings.filterwarnings("ignore")
    if ks is None:
        ks = np.linspace(0, np.pi, 21)
    group_size = None
    if ladder:
        qubits_max_str = f"{n_qubits_max}x2"
        qubits_min_str = f"{n_qubits_max - 1}x2"

    else:
        qubits_max_str = str(n_qubits_max)
        qubits_min_str = str(n_qubits_max - 1)

    vqe_l_dir = f"{qubits_max_str}q_{get_optimal_n_layers(n_qubits_max, ladder, hl)}l"
    vqe_s_dir = (
        f"{qubits_min_str}q_{get_optimal_n_layers(n_qubits_max - 1, ladder, hl)}l"
    )

    # dirty trick
    # now there are multiple names going around
    # it's too late to centralize
    # but we can mitigate
    which_model = model_str(J, hl, ladder)
    model_dir = "vqe_" + which_model

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
        f"{'_'.join(which_algorithms)}_workflow_{model_dir}_hadam_{which_hadamard_test}_{provider_name}_{n_shots}shots_x_{n_reps}reps",
        dry_run=dry_run,
    )

    out_dict = {
        "results": {},
        "inputs": {
            "ks": ks,
            "J": J,
            "hl": hl,
            "ladder": ladder,
            "provider_name": provider_name,
            "group_size": group_size,
            "n_qubits": n_qubits_max,
            "n_shots": n_shots,
            "n_steps": n_steps,
            "n_mc_samples": n_mc_samples,
            "which_algorithms": which_algorithms,
            "which_backends": which_backends,
            "which_hadamard_test": which_hadamard_test,
            "n_reps": n_reps,
            "model_dir": model_dir,
            "postfix": postfix,
            "symmetries": symmetries,
            "which_algorithm_ham": which_algorithm_ham,
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
            hl=hl,
            ladder=ladder,
            symmetries=symmetries,
            postfix=postfix,
            which_algorithm_ham=which_algorithm_ham,
            mitigate_depolarization=mitigate_depolarization,
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
            symmetries=symmetries,
            postfix=postfix,
            which_algorithm_ham=which_algorithm_ham,
            mitigate_depolarization=mitigate_depolarization,
        )

        out_dict["results"]["vqe"] = {
            "dict_l": vqe_dict_l,
            "dict_s": vqe_dict_s,
            "disp": vqe_disp,
        }
        mc_disp = mc_disp_from_res_dict(ks, out_dict, "vqe", n_mc_samples, to_eotm=True)
        out_dict["results"]["vqe"]["mc_disp"] = mc_disp

    ed_disp = tfim_ed_dispersion(n_qubits_max, J, ks, hl=hl, ladder=ladder)
    out_dict["results"]["ed"] = {"disp": {"statevector": ed_disp}}
    edm.save_dict(out_dict)

    edm.var_dump(**out_dict["inputs"])

    fig = plot_all_backends_dispersions_montecarlo(out_dict)

    edm.save_figure(fig, fig_shape="double-y")
    for _b in which_backends:
        show_diffs(out_dict, _b)

    plt.show()

    ...


if __name__ == "__main__":
    style()
    # workflow_nq_aqt_sweep_80j(False, ["real_backend", "fake_backend", "statevector"])
    # workflow_nq_aqt_ladder(False, ["real_backend", "fake_backend", "statevector"])
    # workflow_nq_aqt_hl(False, ["real_backend", "fake_backend", "statevector"])
    # workflow_nq_aqt_j100(False, ["real_backend", "fake_backend", "statevector"])
    # workflow_nq_aqt_j80(False, ["real_backend", "fake_backend", "statevector"])
    # sweep_convergence(False, 0.8, ["statevector"])
    # depol_vs_shots(False, 0.8, ["fake_backend", "statevector"])
    # depol_errmit_vqe(False)
    # workflow_nq_aqt_30j(False, ["real_backend", "fake_backend", "statevector"], "")
    # workflow_nq_aqt_sweep_30j(
    #     True,
    #     ["fake_backend", "statevector"],
    # )
