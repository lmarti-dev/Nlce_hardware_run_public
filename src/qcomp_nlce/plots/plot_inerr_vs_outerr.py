from fauplotstyle import style
import matplotlib.pyplot as plt
from src.qcomp_nlce.utils import (
    BACKEND_LABELS,
    DATA_DIR,
    exact_dispersion_1qp_1d,
    dispersion_1d_thermodynamic,
)
from src.qcomp_nlce.utils.stats import complex_normal, H_effective
from data_manager import ExperimentDataManager
from pathlib import Path
import numpy as np


def plot_errin_vs_errout():
    fpath = "vqe_tfim_80j_0hl_5q"

    dry_run = False
    m_edm = ExperimentDataManager("plot_inerrvsouterr", project="nlce", dry_run=dry_run)

    backend = "statevector"

    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file

    ks = d["inputs"]["ks"]

    name = d["inputs"]["model_dir"]

    true_disp = np.array(d["results"]["vqe"]["disp"]["statevector"])
    expect_l = d["results"]["vqe"]["dict_l"]["exd"]["system"]["values"]
    expect_s = d["results"]["vqe"]["dict_s"]["exd"]["system"]["values"]

    overlap_l = d["results"]["vqe"]["dict_l"]["eld"]["system"]["values"]
    overlap_s = d["results"]["vqe"]["dict_s"]["eld"]["system"]["values"]

    gs_l = d["results"]["vqe"]["dict_l"]["gsd"]["system"]["ground_energy"]
    gs_s = d["results"]["vqe"]["dict_s"]["gsd"]["system"]["ground_energy"]

    true_heff_nb = H_effective(gs_l, expect_l, overlap_l)
    true_heff_ns = H_effective(gs_s, expect_s, overlap_s)

    n_samples = 1000

    errs_in = []
    errs_out = []
    errs_out_std = []
    err = 0.01

    stds_in = []
    disps = np.zeros((len(ks), n_samples))

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

        H_eff_Nb = H_effective(sa_gs_l, sa_expect_l, sa_overlap_l)
        H_eff_Ns = H_effective(sa_gs_s, sa_expect_s, sa_overlap_s)

        return H_eff_Nb, H_eff_Ns

    for n in range(n_samples):
        err += 0.01
        stds_in.append(err)
        print(f"{1 + n:0>6n}/{n_samples}", end="\r")
        H_eff_Nb, H_eff_Ns = sampler_errs(err)
        disp = np.array(
            [dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks]
        )

        err_h_eff_nb = np.mean(np.abs(H_eff_Nb - true_heff_nb))
        err_h_eff_ns = np.mean(np.abs(H_eff_Ns - true_heff_ns))

        errs_in.append(np.mean([err_h_eff_ns, err_h_eff_nb]))
        errs_out.append(np.mean(np.abs(disp - true_disp)))
        errs_out_std.append(np.std(np.abs(disp - true_disp)))

        disps[:, n] = disp

    errinds = np.argsort(errs_in)

    # ax.errorbar(
    #     np.array(errs_in)[errinds],
    #     np.array(errs_out)[errinds],
    #     np.array(errs_out_std)[errinds],
    #     capsize=2,
    #     fmt="x",
    #     linewidth=1,
    # )

    fig, ax = plt.subplots()
    ax.plot(
        np.array(errs_in)[errinds],
        np.array(errs_out)[errinds],
        "x",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlabel(r"$\abs{\tilde{H}_{\mathrm{eff}} -H_{\mathrm{eff}}}$")
    # ax.set_ylabel(r"$\abs{\tilde{\omega}(k) -\omega(k)}$")
    ax.set_xlabel(r"Mean error $|\tilde{H}_{\mathrm{eff}} -H_{\mathrm{eff}}|$")
    ax.set_ylabel(r"Mean error on dispersion")

    m_edm.save_figure(
        fig,
        filename=f"errinout_{n_samples}spl_{name}",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    plt.show()


if __name__ == "__main__":
    style()
    plot_errin_vs_errout()
