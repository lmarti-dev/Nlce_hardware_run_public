from data_manager import ExperimentDataManager
from src.qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
    exact_dispersion_1qp_1d,
    BACKEND_COLORS,
)
from src.qcomp_nlce.utils.results import etopow
import matplotlib.pyplot as plt
from fauplotstyle import style
from src.qcomp_nlce.utils import DATA_DIR
from pathlib import Path


def col_from_fpath(fpath: Path) -> tuple[int, float]:
    keys = ("AQT_BETTER_100", "AQT_WORSE_2", "AQT_WORSE_10", "AQT_BETTER_10", "AQT")

    providers = {
        "AQT_WORSE_10": (0, 10),
        "AQT_WORSE_2": (1, 2),
        "AQT": (2, 1),
        "AQT_BETTER_10": (3, 1 / 10),
        "AQT_BETTER_100": (4, 1 / 100),
    }

    for k in keys:
        if k in fpath:
            return providers[k]


def row_from_nreps(n_shots: int) -> int:
    d = {200: 0, 2000: 1, 10000: 2}
    return d[n_shots]


if __name__ == "__main__":
    style()

    dry_run = True
    m_edm = ExperimentDataManager(
        "plot_depol_vs_shots", project="nlce", dry_run=dry_run
    )

    depol_rate_2q = 0.01
    depol_rate_1q = 0.003

    fpaths = (
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_1reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_50reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_10_200shots_x_1reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_10_200shots_x_10reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_10_200shots_x_50reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_100_200shots_x_1reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_100_200shots_x_10reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_100_200shots_x_50reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_10_200shots_x_1reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_10_200shots_x_10reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_10_200shots_x_50reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_2_200shots_x_1reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_2_200shots_x_10reps",
        r"vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_2_200shots_x_50reps",
    )

    name = "TFIM_vqe_80j_0hl_5q"

    ncols = 5
    nrows = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for fpath in fpaths:
        edm = ExperimentDataManager.load(Path(DATA_DIR, "depol_vs_shots", fpath))
        d = edm.load_last_saved_data_file

        ks = d["inputs"]["ks"]
        j = d["inputs"]["J"]
        n_reps = d["inputs"]["n_reps"]

        fb = d["results"]["vqe"]["disp"]["fake_backend"]
        fbstd = d["results"]["vqe"]["mc_disp"]["fake_backend"]["std"]
        ed = d["results"]["ed"]["disp"]["statevector"]
        ana = exact_dispersion_1qp_1d(j, ks)
        n_shots = d["inputs"]["n_shots"] * d["inputs"]["n_reps"]

        (jj, noise) = col_from_fpath(fpath)
        ii = row_from_nreps(n_shots)

        ax: plt.Axes = axes[ii, jj]

        ax.errorbar(ks, fb, yerr=fbstd, color=BACKEND_COLORS["fake_backend"])
        ax.plot(ks, ana, color=BACKEND_COLORS["analytic"])

        ax.set_yticks([])
        ax.set_xticks([])

        if ii == nrows - 1:
            l1q = depol_rate_2q * noise
            l2q = depol_rate_1q * noise
            xlab = (
                r"\begin{eqnarray*}"
                rf"1q:& {etopow(l1q)}\\"
                rf"2q:& {etopow(l2q)}"
                r"\end{eqnarray*}"
            )
            ax.set_xlabel(xlab, fontsize=8, horizontalalignment="center")
        if jj == 0:
            ylab = f"${etopow(n_shots)}$"
            ax.set_ylabel(ylab, fontsize=8)

    m_edm.var_dump(fpaths=fpaths)
    m_edm.save_figure(
        fig,
        filename=f"{name}_results_depol_study",
        fig_shape="page-wide",
        add_timestamp=False,
    )

    plt.show()
