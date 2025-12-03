from pathlib import Path
from typing import Literal

import io
import json
import copy

import matplotlib.pyplot as plt
import numpy as np
from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    loop_measure_ground_state_energy_subspace_sampling_v2,
)
from qcomp_nlce.hw.quasiparticle.hamiltonian_1qp_expectation import (
    loop_measure_1qp_expectations_v2,
)
from qcomp_nlce.hw.quasiparticle.unitary_1qp_sector import (
    loop_measure_1qp_elements_v2,
)
from qcomp_nlce.utils import (
    TFIM,
    apply_parameters,
    get_min_gap,
    get_tfim_sweep_circuit,
    load_optimized_vqe,
)
from qcomp_nlce.utils.dispersion import (
    dispersion_1d_thermodynamic,
    exact_dispersion_1qp_1d,
)
from qcomp_nlce.utils.effective_hamiltonian import H_effective, H_effective_ed
from qcomp_nlce.utils.folders import res_sim_dir
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from data_manager import ExperimentDataManager

BACKEND_TYPES = Literal["statevector", "fake_backend", "real_backend"]


def dispersion_all_backends(dict_l: dict, dict_s: dict, ks: np.ndarray):
    disp = {}
    for which in dict_l["heff"].keys():
        H_eff_Nb = dict_l["heff"][which]
        H_eff_Ns = dict_s["heff"][which]
        disp[which] = [dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks]
    return disp


def vqe_dispersion(
    gs_dirname_l: Path,
    elem_dirname_l: Path,
    gs_dirname_s: Path,
    elem_dirname_s: Path,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    ks: np.ndarray,
    group_size: int,
    which_hadamard_test: str,
    uncertainties: bool,
):
    dict_l = vqe_heff(
        gs_dirname_l,
        elem_dirname_l,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        uncertainties,
    )
    dict_s = vqe_heff(
        gs_dirname_s,
        elem_dirname_s,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        uncertainties,
    )

    return dict_l, dict_s, dispersion_all_backends(dict_l, dict_s, ks)


def sweep_dispersion(
    n_qubits_max: int,
    J: float,
    mult: float,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    ks: np.ndarray,
    group_size: int,
    which_hadamard_test: str,
    uncertainties: bool,
):
    dict_l = sweep_heff(
        n_qubits_max,
        J,
        mult,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        uncertainties,
    )
    dict_s = sweep_heff(
        n_qubits_max - 1,
        J,
        mult,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        uncertainties,
    )

    return dict_l, dict_s, dispersion_all_backends(dict_l, dict_s, ks)


def vqe_heff(
    gs_dirname: Path,
    elem_dirname: Path,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    group_size: int,
    which_hadamard_test: str,
    uncertainties: bool,
) -> dict:

    hamiltonian, elem_circuit, elem_weights, ground_energy = load_optimized_vqe(
        elem_dirname
    )
    elem_circuit = apply_parameters(elem_circuit, elem_weights[-1])
    hamiltonian, gs_circuit, gs_weights, ground_energy = load_optimized_vqe(gs_dirname)
    gs_circuit = apply_parameters(gs_circuit, gs_weights[-1])
    assert hamiltonian == hamiltonian

    return get_heff_values(
        gs_circuit,
        elem_circuit,
        hamiltonian,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        uncertainties,
    )


def sweep_heff(
    n_qubits: int,
    J: float,
    mult: float,
    which_backends: list[str],
    provider_name: str,
    n_shots: int,
    group_size: int,
    which_hadamard_test: str,
    uncertainties: bool,
):
    ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = mult / (min_gap**2)
    n_steps = int(mult * n_qubits**2)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )
    return get_heff_values(
        circuit,
        circuit,
        hamiltonian,
        which_backends,
        provider_name,
        n_shots,
        group_size,
        which_hadamard_test,
        uncertainties,
    )


def load_and_plot(fpath: Path):
    with io.open(fpath, "r", encoding="utf8") as f:
        d = json.loads(f.read())
    plot_all_dispersions(d)


def heff_backend_type(eld: dict, gsd: dict, exd: dict, which_backend: BACKEND_TYPES):
    overlap = eld[which_backend]["values"]
    one_qp_expectation = exd[which_backend]["values"]
    gs_expectation = gsd[which_backend]["energy"]

    heff = H_effective(gs_expectation, one_qp_expectation, overlap)

    return heff


def get_heff_values(
    gs_circuit: QuantumCircuit,
    elem_circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    which_backends: list[str],
    provider_name: str = "IBM",
    n_shots: int = 1000,
    group_size: int = None,
    which_hadamard_test: str = "full",
    solve: bool = True,
    uncertainties: bool = False,
):

    print("-- Computing <j|U|k> --")
    elements_dict = loop_measure_1qp_elements_v2(
        circuit=elem_circuit,
        which_backends=which_backends,
        solve=solve,
        provider_name=provider_name,
        n_shots=n_shots,
        group_size=group_size,
        which_hadamard_test=which_hadamard_test,
    )

    print("-- Computing ground state energy --")
    ground_state_dict = loop_measure_ground_state_energy_subspace_sampling_v2(
        hamiltonian=hamiltonian,
        circuit=gs_circuit,
        which_backends=which_backends,
        n_shots=n_shots,
        provider_name=provider_name,
        solve=solve,
    )

    print("-- Computing <j|U* H U|k> --")
    expectations_dict = loop_measure_1qp_expectations_v2(
        hamiltonian=hamiltonian,
        circuit=elem_circuit,
        which_backends=which_backends,
        provider_name=provider_name,
        n_shots=n_shots,
        solve=solve,
    )

    heff = {}

    for which_backend in which_backends:
        heff[which_backend] = heff_backend_type(
            elements_dict,
            ground_state_dict,
            expectations_dict,
            which_backend,
        )

    return {
        "eld": elements_dict,
        "gsd": ground_state_dict,
        "exd": expectations_dict,
        "heff": heff,
    }


def tfim_ed_dispersion(n_qubits_max: int, J: float, ks: np.ndarray):
    hamiltonian_l = TFIM(n_qubits_max, 1, J)
    hamiltonian_s = TFIM(n_qubits_max - 1, 1, J)
    return ed_dispersion(hamiltonian_l, hamiltonian_s, ks)


def ed_dispersion(
    hamiltonian_l: SparsePauliOp, hamiltonian_s: SparsePauliOp, ks: np.ndarray
) -> list[float]:
    heff_l = H_effective_ed(hamiltonian_l)
    heff_s = H_effective_ed(hamiltonian_s)
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


def plot_all_dispersions(_d: dict) -> plt.Figure:
    ks = _d["inputs"]["ks"]
    j = _d["inputs"]["J"]
    fig, ax = plt.subplots()
    res = _d["results"]
    for k in res:
        disp = res[k]["disp"]
        for be in disp:
            ax.plot(ks, disp[be], label=f"{k} - {be}")
    ax.plot(ks, exact_dispersion_1qp_1d(j, ks), label="analytic")
    ax.legend()
    return fig


def plot_multiple_dispersions(_l: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots()
    for ind, _d in enumerate(_l):
        ks = _d["inputs"]["ks"]
        j = _d["inputs"]["J"]
        res = _d["results"]
        for k in res:
            disp = res[k]["disp"]
            for be in disp:
                ax.plot(ks, disp[be], label=f"{k} - {be} ({ind})")
    ax.plot(ks, exact_dispersion_1qp_1d(j, ks), label="analytic")
    ax.legend()
    return fig


def u_plot_all_dispersions(d: dict) -> plt.Figure:
    ks = d["inputs"]["ks"]
    j = d["inputs"]["J"]
    fig, ax = plt.subplots()
    res = d["results"]
    for k in res:
        disp = res[k]["disp"]
        for be in disp:

            ax.errorbar(ks, disp[be][0][:], yerr=disp[be][1][:], label=f"{k} - {be}")
            ax.legend()

    ax.plot(ks, exact_dispersion_1qp_1d(j, ks), label="analytic")
    ax.legend()
    return fig


def workflow_5q():

    n_qubits_max = 5
    J = 1
    mult = 1
    which_backends = [
        "statevector",
        "fake_backend",
    ]
    provider_name = "AQT"
    n_shots = int(2e3)
    ks = np.linspace(0, np.pi, 21)
    group_size = 6
    which_hadamard_test = "full"
    uncertainties = False

    gs_dirname_l = Path(res_sim_dir(), "gs_vqe_1j/vqe/5q_3l")
    gs_dirname_s = Path(res_sim_dir(), "gs_vqe_1j/vqe/4q_2l")
    elem_dirname_l = Path(res_sim_dir(), "elem_vqe_1j/vqe/cost_offdiag/5q_3l")
    elem_dirname_s = Path(res_sim_dir(), "elem_vqe_1j/vqe/cost_offdiag/4q_2l")

    dry_run = False

    edm = ExperimentDataManager("workflow_tfim_all_backends_test", dry_run=dry_run)

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
    # uncertainties,
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
        uncertainties,
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
    # plt.show()


def average_matrices(dicts: list[dict]) -> dict:

    n_qubits = 5
    n_dicts = len(dicts)

    out_d = copy.deepcopy(dicts[0])
    ks = out_d["inputs"]["ks"]
    for si in ("dict_l", "dict_s"):
        for backend in out_d["results"]["vqe"][si]["eld"]:
            if backend != "system":
                out_d["results"]["vqe"][si]["eld"][backend]["values"] = (
                    out_d["results"]["vqe"][si]["eld"][backend]["values"] / n_dicts
                )
                out_d["results"]["vqe"][si]["exd"][backend]["values"] = (
                    out_d["results"]["vqe"][si]["exd"][backend]["values"] / n_dicts
                )
                out_d["results"]["vqe"][si]["gsd"][backend]["energy"] = (
                    out_d["results"]["vqe"][si]["gsd"][backend]["energy"] / n_dicts
                )

    for ii in range(1, len(dicts)):
        d = dicts[ii]
        for si in ("dict_l", "dict_s"):
            for backend in d["results"]["vqe"][si]["eld"]:
                if backend != "system":
                    out_d["results"]["vqe"][si]["eld"][backend]["values"] = (
                        out_d["results"]["vqe"][si]["eld"][backend]["values"]
                        + d["results"]["vqe"][si]["eld"][backend]["values"] / n_dicts
                    )
                    out_d["results"]["vqe"][si]["exd"][backend]["values"] = (
                        out_d["results"]["vqe"][si]["exd"][backend]["values"]
                        + d["results"]["vqe"][si]["exd"][backend]["values"] / n_dicts
                    )
                    out_d["results"]["vqe"][si]["gsd"][backend]["energy"] = (
                        out_d["results"]["vqe"][si]["gsd"][backend]["energy"]
                        + d["results"]["vqe"][si]["gsd"][backend]["energy"] / n_dicts
                    )
    for si in ("dict_l", "dict_s"):
        for backend in d["results"]["vqe"][si]["eld"]:
            if backend != "system":
                eld = out_d["results"]["vqe"][si]["eld"]
                gsd = out_d["results"]["vqe"][si]["gsd"]
                exd = out_d["results"]["vqe"][si]["exd"]
                heff = heff_backend_type(eld, gsd, exd, backend)
                out_d["results"]["vqe"][si]["heff"][backend] = heff
        disp = dispersion_all_backends(
            out_d["results"]["vqe"]["dict_l"], out_d["results"]["vqe"]["dict_s"], ks
        )
    out_d["results"]["vqe"]["disp"] = disp
    return out_d


def average_over_runs():
    runs = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00005",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00001",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00002",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00003",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00004",
    )

    dicts = []
    for r in runs:
        edm = ExperimentDataManager.load(r)
        d = edm.load_last_saved_data_file
        dicts.append(d)

    out_d = average_matrices(dicts)

    fig = plot_multiple_dispersions([out_d, dicts[0]])

    plt.show()


if __name__ == "__main__":
    workflow_5q()
