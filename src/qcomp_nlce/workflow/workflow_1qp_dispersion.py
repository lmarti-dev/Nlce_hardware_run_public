from pathlib import Path

import numpy as np

from qcomp_nlce.utils.folders import res_sim_dir
from qcomp_nlce.utils.workflow import (
    vqe_dispersion,
    tfim_ed_dispersion,
    sweep_dispersion,
    plot_all_backends_dispersions_UT,
)
from data_manager import ExperimentDataManager

import matplotlib.pyplot as plt
import warnings


def workflow_5q():
    # avoid the "unused arguments" warning
    # warnings.filterwarnings("ignore")
    n_qubits_max = 5
    J = 0.8
    which_backends = [
        "fake_backend",
        "statevector",
    ]
    provider_name = "AQT"
    n_shots = int(2e3)
    ks = np.linspace(0, np.pi, 21)
    group_size = None
    which_hadamard_test = "stefan"

    gs_dirname_l = Path(res_sim_dir(), f"gs_vqe_{int(J*100)}j/vqe/5q_3l")
    gs_dirname_s = Path(res_sim_dir(), f"gs_vqe_{int(J*100)}j/vqe/4q_2l")
    elem_dirname_l = Path(res_sim_dir(), f"elem_vqe_{int(J*100)}j/vqe/cost_offdiag/5q_3l")
    elem_dirname_s = Path(res_sim_dir(), f"elem_vqe_{int(J*100)}j/vqe/cost_offdiag/4q_2l")

    dry_run = True

    edm = ExperimentDataManager(
        f"workflow_tfim_hadam_{which_hadamard_test}_{provider_name}_{n_shots}shots",
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
        },
    }

    # print("== Sweep ==")
    # sweep_dict_l, sweep_dict_s, sweep_disp = sweep_dispersion(
    #     n_qubits_max,
    #     J,
    #     mult,
    #     which_backends,
    #     provider_name,
    #     n_shots,
    #     ks,
    #     group_size,
    #     which_hadamard_test,
    # )

    # out_dict["results"]["sweep"] = {
    #     "dict_l": sweep_dict_l,
    #     "dict_s": sweep_dict_s,
    #     "disp": sweep_disp,
    # }

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

    # Print effective Hamiltonians
    for backend in which_backends:
        print(f"\nH_eff (large, {backend}):\n{vqe_dict_l['heff'][backend]}")
        print(f"\nH_eff (small, {backend}):\n{vqe_dict_s['heff'][backend]}")

    out_dict["results"]["vqe"] = {
        "dict_l": vqe_dict_l,
        "dict_s": vqe_dict_s,
        "disp": vqe_disp,
    }

    ed_disp = tfim_ed_dispersion(n_qubits_max, J, ks)
    out_dict["results"]["ed"] = {"disp": {"statevector": ed_disp}}
    edm.save_dict(out_dict)

    fig = plot_all_backends_dispersions_UT(out_dict)

    edm.save_figure(fig, fig_shape="double-y")
    plt.show()


def worflow_sweep_test():

    for n_steps in range(4, 30, 5):
        n_qubits_max = 5
        J = 1
        which_backends = ["statevector", "fake_backend"]
        provider_name = "AQT"
        n_shots = int(2e3)
        ks = np.linspace(0, np.pi, 21)
        group_size = None
        which_hadamard_test = "stefan"

        dry_run = False

        edm = ExperimentDataManager(
            f"workflow_tfim_sweep_nsteps_{provider_name}_{n_shots}shots",
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
                "n_steps": n_steps,
            },
        }

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

        ed_disp = tfim_ed_dispersion(n_qubits_max, J, ks)
        out_dict["results"]["ed"] = {"disp": {"statevector": ed_disp}}
        edm.save_dict(out_dict)

        edm.var_dump(n_steps=n_steps, provider_name=provider_name)

        fig = plot_all_backends_dispersions_UT(out_dict)

        edm.save_figure(fig, fig_shape="double-y")


if __name__ == "__main__":
    workflow_5q()
