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
import os

def col_from_fpath(fpath: Path) -> tuple[int, float]:
    keys = ("AQT_BETTER_100", "AQT_WORSE_2", "AQT_WORSE_10", "AQT_BETTER_10", "AQT")

    providers = {
        # "AQT_WORSE_10": (0, 10),
        "AQT_WORSE_2": (0, 2),
        "AQT": (1, 1),
        "AQT_BETTER_10": (2, 1 / 10),
        "AQT_BETTER_100": (3, 1 / 100),
    }

    for k in keys:
        if k in fpath.as_posix():
            return providers[k]


def row_from_nreps(n_shots: int) -> int:
    d = {200: 0, 2000: 1, 10000: 2, 40000: 3}
    return d[n_shots]


def plot_4x4_from_fpaths(
    fpaths: list[Path],
    name: str,
    depol_rate_1q: float,
    depol_rate_2q: float,
    dry_run: bool = True,
):

    m_edm = ExperimentDataManager(
        f"plot_depol_vs_shots_{name}", project="nlce", dry_run=dry_run
    )

    ncols = 4
    nrows = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for fpath in fpaths:
        edm = ExperimentDataManager.load(fpath)
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
            l1q = depol_rate_1q * noise
            l2q = depol_rate_2q * noise
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


def plot_tfim_80j(dry_run):    
    style()


    depol_rate_2q = 0.01
    depol_rate_1q = 0.003

    dirname = Path(DATA_DIR,"depol_vs_shots_80j")
    fpaths = [Path(dirname,f) for f in os.listdir(dirname)]

    name = "TFIM_vqe_80j_0hl_5q"

    plot_4x4_from_fpaths(fpaths, name, depol_rate_1q, depol_rate_2q, dry_run)



def plot_tfim_50j_10hl(dry_run):
    style()

    depol_rate_2q = 0.01
    depol_rate_1q = 0.003

    dirname = Path(DATA_DIR,"depol_vs_shots_50j_10hl")
    fpaths = [Path(dirname,f) for f in os.listdir(dirname)]

    name = "TFIM_vqe_50j_10hl_5q"

    plot_4x4_from_fpaths(fpaths, name, depol_rate_1q, depol_rate_2q, dry_run)


if __name__ == "__main__":

    dry_run = True

    plot_tfim_50j_10hl(dry_run)