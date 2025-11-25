from qcomp_nlce.hw.quasiparticle.unitary_1qp_sector import measure_1qp_elements
from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    measure_ground_state_energy_subspace_sampling,
)

from Nlce_hardware_run.src.qcomp_nlce.hw.quasiparticle.hamiltonian_1qp_expectation import (
    measure_1qp_expectations,
)

from qcomp_nlce.utils import (
    load_optimized_vqe,
    TFIM,
    get_min_gap,
    apply_parameters,
    get_tfim_sweep_circuit,
    get_ground_energy,
)
from qcomp_nlce.utils.dispersion import (
    dispersion_1d_thermodynamic,
    exact_dispersion_1qp_1d,
)
from qcomp_nlce.utils.effective_hamiltonian import H_effective, H_effective_ed

from qcomp_nlce.utils.folders import res_sim_dir

from pathlib import Path
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from typing import Literal, get_args
from data_manager import ExperimentDataManager

import numpy as np

import matplotlib.pyplot as plt

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
    run_on_hardware: bool,
    backend_name: str,
    n_shots: int,
    ks: np.ndarray,
):
    dict_l = vqe_heff(
        gs_dirname_l,
        elem_dirname_l,
        run_on_hardware,
        backend_name,
        n_shots,
    )
    dict_s = vqe_heff(
        gs_dirname_s,
        elem_dirname_s,
        run_on_hardware,
        backend_name,
        n_shots,
    )

    return dict_l, dict_s, dispersion_all_backends(dict_l, dict_s, ks)


def sweep_dispersion(
    n_qubits_max: int,
    J: float,
    mult: float,
    run_on_hardware: bool,
    backend_name: str,
    n_shots: int,
    ks: np.ndarray,
):
    dict_l = sweep_heff(n_qubits_max, J, mult, run_on_hardware, backend_name, n_shots)
    dict_s = sweep_heff(
        n_qubits_max - 1, J, mult, run_on_hardware, backend_name, n_shots
    )

    return dict_l, dict_s, dispersion_all_backends(dict_l, dict_s, ks)


def vqe_heff(
    gs_dirname: Path,
    elem_dirname: Path,
    run_on_hardware: bool,
    backend_name: str,
    n_shots: int,
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
        ground_energy,
        run_on_hardware,
        backend_name,
        n_shots,
    )


def sweep_heff(
    n_qubits: int,
    J: float,
    mult: float,
    run_on_hardware: bool,
    backend_name: str,
    n_shots: int,
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
    ground_energy = get_ground_energy(hamiltonian)
    return get_heff_values(
        circuit,
        circuit,
        hamiltonian,
        ground_energy,
        run_on_hardware,
        backend_name,
        n_shots,
        1,
    )


def heff_backend_type(
    eld: dict,
    gsd: dict,
    exd: dict,
    which: BACKEND_TYPES,
):
    overlap = eld[which]["values"]
    one_qp_expectation = exd[which]["values"]
    gs_expectation = gsd[which]["energy"]

    heff = H_effective(gs_expectation, one_qp_expectation, overlap)

    return heff


def get_heff_values(
    gs_circuit: QuantumCircuit,
    elem_circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    run_on_hardware: bool,
    backend_name: str = "IBM",
    n_shots: int = 1000,
    group_size: int = None,
):
    elements_dict = measure_1qp_elements(
        elem_circuit, backend_name, n_shots, run_on_hardware, group_size
    )

    ground_state_dict = measure_ground_state_energy_subspace_sampling(
        hamiltonian, ground_energy, gs_circuit, n_shots, run_on_hardware, backend_name
    )

    expectations_dict = measure_1qp_expectations(
        hamiltonian, elem_circuit, backend_name, n_shots, run_on_hardware
    )

    heff = {}

    for which in get_args(BACKEND_TYPES):
        if not run_on_hardware and which == "real_backend":
            pass
        else:
            heff[which] = heff_backend_type(
                elements_dict, ground_state_dict, expectations_dict, which
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


def plot_all_dispersions(d: dict) -> plt.Figure:
    ks = d["inputs"]["ks"]
    j = d["inputs"]["J"]
    fig, ax = plt.subplots()
    res = d["results"]
    for k in res:
        disp = res[k]["disp"]
        for be in disp:
            ax.plot(ks, disp[be], label=f"{k} - {be}")
    ax.plot(ks, exact_dispersion_1qp_1d(j, ks), label="analytic")
    ax.legend()
    return fig


if __name__ == "__main__":

    n_qubits_max = 5
    J = 1
    mult = 0.5
    run_on_hardware = False
    backend_name = "AQT"
    n_shots = int(1e4)
    ks = np.linspace(0, np.pi, 21)
    gs_dirname_l = Path(res_sim_dir(), "gs_vqe_1j/vqe/5q_3l")
    elem_dirname_l = Path(res_sim_dir(), "elem_vqe_1j/vqe/5q_3l")
    gs_dirname_s = Path(res_sim_dir(), "gs_vqe_1j/vqe/4q_2l")
    elem_dirname_s = Path(res_sim_dir(), "elem_vqe_1j/vqe/4q_2l")

    edm = ExperimentDataManager("workflow_tfim_all_backends", dry_run=True)

    out_dict = {
        "results": {},
        "inputs": {"ks": ks, "J": J, "backend_name": backend_name},
    }

    sweep_dict_l, sweep_dict_s, sweep_disp = sweep_dispersion(
        n_qubits_max, J, mult, run_on_hardware, backend_name, n_shots, ks
    )

    out_dict["results"]["sweep"] = {
        "dict_l": sweep_dict_l,
        "dict_s": sweep_dict_s,
        "disp": sweep_disp,
    }

    # vqe_dict_l, vqe_dict_s, vqe_disp = vqe_dispersion(
    #     gs_dirname_l,
    #     elem_dirname_l,
    #     gs_dirname_s,
    #     elem_dirname_s,
    #     run_on_hardware,
    #     backend_name,
    #     n_shots,
    #     ks,
    # )

    # out_dict["results"]["vqe"] = {
    #     "dict_l": vqe_dict_l,
    #     "dict_s": vqe_dict_s,
    #     "disp": vqe_disp,
    # }

    ed_disp = tfim_ed_dispersion(n_qubits_max, J, ks)
    out_dict["results"]["ed"] = {"disp": {"statevector": ed_disp}}

    fig = plot_all_dispersions(out_dict)

    plt.show()

    edm.save_dict(out_dict)
    edm.save_figure(fig, fig_shape="double-y")
