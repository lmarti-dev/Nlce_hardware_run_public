from pathlib import Path
from typing import Literal


import numpy as np

from qcomp_nlce.utils.folders import res_sim_dir
from qcomp_nlce.utils.workflow import (
    vqe_dispersion,
    tfim_ed_dispersion,
    plot_all_dispersions,
)
import matplotlib.pyplot as plt
from data_manager import ExperimentDataManager


def workflow_5q():

    n_qubits_max = 5
    J = 1
    mult = 1
    which_backends = [
        "fake_backend",
    ]
    provider_name = "AQT"
    n_shots = int(2e3)
    ks = np.linspace(0, np.pi, 21)
    group_size = 6
    which_hadamard_test = "estimator"

    gs_dirname_l = Path(res_sim_dir(), "gs_vqe_1j/vqe/5q_3l")
    gs_dirname_s = Path(res_sim_dir(), "gs_vqe_1j/vqe/4q_2l")
    elem_dirname_l = Path(res_sim_dir(), "elem_vqe_1j/vqe/cost_offdiag/5q_3l")
    elem_dirname_s = Path(res_sim_dir(), "elem_vqe_1j/vqe/cost_offdiag/4q_2l")

    dry_run = False

    edm = ExperimentDataManager(
        "workflow_tfim_real_backend_vqe_REALDATA", dry_run=dry_run
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

    print("== Sweep ==")
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

    fig = plot_all_dispersions(out_dict)

    edm.save_figure(fig, fig_shape="double-y")
    plt.show()


if __name__ == "__main__":
    workflow_5q()
