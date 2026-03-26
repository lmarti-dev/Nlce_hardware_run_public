from fauplotstyle import style
import matplotlib.pyplot as plt
from src.qcomp_nlce.utils import (
    BACKEND_LABELS,
    DATA_DIR,
    exact_dispersion_1qp_1d,
    dispersion_1d_thermodynamic,
    projector_ham,
    TFIM,
    BACKEND_COLORS,
)


from src.qcomp_nlce.utils.workflow_utils import tfim_ed_dispersion

from src.qcomp_nlce.utils.stats import complex_normal, H_effective
from data_manager import ExperimentDataManager
from pathlib import Path
import numpy as np
from typing import Literal
from qiskit.quantum_info import SparsePauliOp


def tr_cx(n_qubits: int, which: Literal["X", "Y"]) -> float:

    proj = projector_ham(n_qubits)
    ham = proj ^ SparsePauliOp(which)

    return float(np.trace(ham.to_matrix()))


def depol_overlap(o: np.ndarray, p: float, n_qubits: int):
    r = o.real
    i = o.imag

    trr = tr_cx(n_qubits, "X")
    tri = tr_cx(n_qubits, "Y")

    return depolarize(r, trr, p, n_qubits) + 1j * depolarize(i, tri, p, n_qubits)


def depol_hamelem(o: complex, p: float, trham: float, diag: bool, n_qubits: int):

    if diag:
        return depolarize(o, trham, p, n_qubits)
    else:
        # since it's approx the same circuit, off diag depol cancels out
        # because gamma1 - gamma2 are ish the same
        return (1 - p) * o


def depol_expect(o: np.ndarray, p: float, trham: float, n_qubits: int):
    out = np.zeros_like(o)
    for ii in range(o.shape[0]):
        for jj in range(o.shape[1]):
            out[ii, jj] = depol_hamelem(
                o[ii, jj], p, trham, diag=ii == jj, n_qubits=n_qubits
            )

    return out


def depolarize(o: np.ndarray, tro: float, p: float, n_qubits: int) -> np.ndarray:
    print(f"trace: {tro}")
    if isinstance(o, np.ndarray):
        p = p + 1e-1 * (np.random.rand(*o.shape) - 0.5)
    else:
        p = p + 1e-1 * (np.random.rand() - 0.5)

    return (1 - p) * o + p * tro / (2**n_qubits)


def plot_depol_effect(fpath, dry_run):

    m_edm = ExperimentDataManager("plot_effect_depol", project="nlce", dry_run=dry_run)

    backend = "statevector"

    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file

    ks = d["inputs"]["ks"]
    name = d["inputs"]["model_dir"]
    j = d["inputs"]["J"]
    hl = d["inputs"]["hl"]
    ladder = d["inputs"]["ladder"]
    n_qubits = d["inputs"]["n_qubits"]

    ham_l = TFIM(n_qubits, 1, j, longitudinal_field=hl, ladder=ladder)
    ham_s = TFIM(n_qubits - 1, 1, j, longitudinal_field=hl, ladder=ladder)

    trham_l = np.trace(ham_l.to_matrix())
    trham_s = np.trace(ham_s.to_matrix())

    expect_l = d["results"]["vqe"]["dict_l"]["exd"]["system"]["values"]
    expect_s = d["results"]["vqe"]["dict_s"]["exd"]["system"]["values"]

    overlap_l = d["results"]["vqe"]["dict_l"]["eld"]["system"]["values"]
    overlap_s = d["results"]["vqe"]["dict_s"]["eld"]["system"]["values"]

    gs_l = d["results"]["vqe"]["dict_l"]["gsd"]["system"]["ground_energy"]
    gs_s = d["results"]["vqe"]["dict_s"]["gsd"]["system"]["ground_energy"]

    n_samples = 100

    n_shots = 200

    depol_pows = np.linspace(-5, -1, 3)
    disps = np.zeros((len(depol_pows), len(ks), n_samples))

    def sampler_errs(err):
        sa_expect_l = complex_normal(
            loc=expect_l,
            scale=(1 + 1j) * err,
        )
        sa_overlap_l = complex_normal(
            loc=overlap_l,
            scale=(1 + 1j) * err,
        )
        sa_gs_l = complex_normal(
            loc=gs_l,
            scale=(1 + 1j) * err,
        )
        sa_expect_s = complex_normal(
            loc=expect_s,
            scale=(1 + 1j) * err,
        )
        sa_overlap_s = complex_normal(
            loc=overlap_s,
            scale=(1 + 1j) * err,
        )
        sa_gs_s = complex_normal(
            loc=gs_s,
            scale=(1 + 1j) * err,
        )
        return sa_expect_l, sa_overlap_l, sa_gs_l, sa_expect_s, sa_overlap_s, sa_gs_s

    def depoledheff(depol):

        err_expect_l, err_overlap_l, err_gs_l, err_expect_s, err_overlap_s, err_gs_s = (
            sampler_errs(1 / np.sqrt(n_shots))
        )

        sa_expect_l = depol_expect(err_expect_l, depol, trham_l, n_qubits)
        sa_expect_s = depol_expect(err_expect_s, depol, trham_s, n_qubits)
        sa_overlap_l = depol_overlap(err_overlap_l, depol, n_qubits)
        sa_overlap_s = depol_overlap(err_overlap_s, depol, n_qubits)
        sa_gs_l = depolarize(err_gs_l, trham_l, depol, n_qubits)
        sa_gs_s = depolarize(err_gs_s, trham_s, depol, n_qubits)

        H_eff_Nb = H_effective(sa_gs_l, sa_expect_l, sa_overlap_l)
        H_eff_Ns = H_effective(sa_gs_s, sa_expect_s, sa_overlap_s)

        return H_eff_Nb, H_eff_Ns

    for jj, depol_pow in enumerate(depol_pows):
        depol = 10**depol_pow
        for n in range(n_samples):
            print(f"{1 + n:0>6n}/{n_samples}", end="\r")
            H_eff_Nb, H_eff_Ns = depoledheff(depol)
            disp = [dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks]
            disps[jj, :, n] = disp

    meaned_disps = np.mean(disps, axis=2)
    meaned_disps_std = np.std(disps, axis=2)

    fig, ax = plt.subplots()
    for jj, depol_pow in enumerate(depol_pows):
        ax.errorbar(
            ks,
            meaned_disps[jj, :],
            meaned_disps_std[jj, :],
            capsize=2,
            fmt="x:",
            linewidth=1,
            label=f"$10^{{{depol_pow:n}}}$",
        )
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(r"$\omega(k)$")
    ax.set_ylabel(r"$k$")

    if "_0hl" in name and "ladder" not in name:
        ax.plot(
            ks,
            exact_dispersion_1qp_1d(j, ks),
            label="Analytic",
            color=BACKEND_COLORS["analytic"],
        )
    ax.plot(
        ks,
        tfim_ed_dispersion(n_qubits, j, ks, hl, ladder),
        label=BACKEND_LABELS["ed"],
        color=BACKEND_COLORS["ed"],
    )

    ax.legend()

    m_edm.save_figure(
        fig,
        filename=f"depol_effect_{n_samples}spl_{name}",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    plt.show()


if __name__ == "__main__":
    style()
    fpath = "1qp/vqe_tfim_80j_0hl_5q"
    plot_depol_effect(fpath, True)
