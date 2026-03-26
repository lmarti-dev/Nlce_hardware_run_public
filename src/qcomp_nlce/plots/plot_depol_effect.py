from fauplotstyle import style
import matplotlib.pyplot as plt
from src.qcomp_nlce.utils import (
    BACKEND_LABELS,
    DATA_DIR,
    exact_dispersion_1qp_1d,
    dispersion_1d_thermodynamic,
    projector_ham,
)
from src.qcomp_nlce.utils.stats import complex_normal, H_effective
from data_manager import ExperimentDataManager
from pathlib import Path
import numpy as np
from typing import Literal
from qiskit.quantum_info import SparsePauliOp


def tr_cx(n_qubits: int, which: Literal["X", "Y"]):

    proj = projector_ham(n_qubits)
    ham = proj ^ SparsePauliOp(which)

    return np.trace(ham.to_matrix())


def depol_cx(o: np.ndarray, p: float, n_qubits: int):
    r = o.real
    i = o.imag

    trr = tr_cx(n_qubits, "X")
    tri = tr_cx(n_qubits, "Y")

    return depolarize(r, trr, p, n_qubits) + 1j * depolarize(i, tri, p, n_qubits)


def depol_ham(o: complex, p, ham: SparsePauliOp, diag: bool):
    n_qubits = ham.num_qubits
    hammat = ham.to_matrix()

    o


def depolarize(o: np.ndarray, tro: float, p: float, n_qubits: int) -> np.ndarray:
    return (1 - p) * o + p * tro / (2**n_qubits)


def plot_errin_vs_errout():
    fpath = "vqe_tfim_80j_0hl_5q"

    dry_run = False
    m_edm = ExperimentDataManager(
        "plot_instd_vs_out_std", project="nlce", dry_run=dry_run
    )

    backend = "statevector"

    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file

    ks = d["inputs"]["ks"]
    name = d["inputs"]["model_dir"]

    expect_l = d["results"]["vqe"]["dict_l"]["exd"]["system"]["values"]
    expect_s = d["results"]["vqe"]["dict_s"]["exd"]["system"]["values"]

    overlap_l = d["results"]["vqe"]["dict_l"]["eld"]["system"]["values"]
    overlap_s = d["results"]["vqe"]["dict_s"]["eld"]["system"]["values"]

    gs_l = d["results"]["vqe"]["dict_l"]["gsd"]["system"]["ground_energy"]
    gs_s = d["results"]["vqe"]["dict_s"]["gsd"]["system"]["ground_energy"]

    n_samples = 10000

    std_pows = np.arange(-5, 0)

    stds_in = []
    disps = np.zeros((len(std_pows), len(ks), n_samples))

    for depol_pow in np.linspace(-5, 1, 6):
        depol = 10**depol_pow
        sa_expect_l = depolarize(expect_l, trex)

        H_eff_Nb = H_effective(sa_gs_l, sa_expect_l, sa_overlap_l)
        H_eff_Ns = H_effective(sa_gs_s, sa_expect_s, sa_overlap_s)

        return H_eff_Nb, H_eff_Ns

    for jj, std_pow in enumerate(std_pows):
        std = 10.0**std_pow
        stds_in.append(std)
        for n in range(n_samples):
            print(f"{1 + n:0>6n}/{n_samples}", end="\r")
            H_eff_Nb, H_eff_Ns = sampler_errs(std)
            disp = [dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks]
            disps[jj, :, n] = disp

    stds_out = np.mean(np.std(disps, axis=2), axis=1)
    stds_out_std = np.std(np.std(disps, axis=2), axis=1)

    fig, ax = plt.subplots()
    ax.errorbar(
        stds_in,
        stds_out,
        stds_out_std,
        capsize=2,
        fmt="x:",
        linewidth=1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Standard dev. of $\tilde{H}_{\mathrm{eff}}$")
    ax.set_ylabel("Mean standard dev. of dispersion")

    m_edm.save_figure(
        fig,
        filename=f"stdinout_{n_samples}spl_{name}",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    plt.show()


if __name__ == "__main__":
    style()
    plot_errin_vs_errout()
