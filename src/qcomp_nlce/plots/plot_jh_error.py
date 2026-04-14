import matplotlib.pyplot as plt
from data_manager import ExperimentDataManager

from src.qcomp_nlce.utils import DATA_DIR

from pathlib import Path
from fauplotstyle import style
import numpy as np


def getcolor(fpath):
    if "ladder" in fpath:
        return "#971B2F"
    elif "_0hl" in fpath:
        return "#228848"
    else:
        return "#E87722"


if __name__ == "__main__":
    style()
    fig, ax = plt.subplots()

    ALGSYM = {"sweep": "o", "vqe": "x"}

    dry_run = False

    edm = ExperimentDataManager("plot_all_errors", project="nlce", dry_run=dry_run)

    handles = {}

    paths = (
        ("vqe_tfim_30j_0hl_5q", "TFIM_30j_0hl", "vqe"),
        ("vqe_tfim_80j_0hl_5q", "TFIM_80j_0hl", "vqe"),
        ("vqe_tfim_80j_0hl_5q_run2", "TFIM_80j_0hl", "vqe"),
        ("vqe_tfim_50j_10hl_5q", "TFIM_50j_10hl", "vqe"),
        ("vqe_tfim_50j_10hl_5q_run2", "TFIM_50j_10hl", "vqe"),
        ("vqe_tfim_50j_10hl_5q_run3", "TFIM_50j_10hl", "vqe"),
        ("sweep_tfim_30j_0hl_5q", "TFIM_30j_0hl", "sweep"),
        ("sweep_tfim_80j_0hl_5q", "TFIM_80j_0hl", "sweep"),
        ("sweep_tfim_80j_0hl_5q_run2", "TFIM_80j_0hl", "sweep"),
        ("vqe_tfim_100j_0hl_5q", "TFIM_100j_0hl", "vqe"),
        ("vqe_tfim_100j_0hl_5q_run2", "TFIM_100j_0hl", "vqe"),
        ("vqe_tfim_100j_0hl_5q_run3", "TFIM_100j_0hl", "vqe"),
        ("vqe_tfim_20j_0hl_8q_ladder", "TFIM_20j_0hl_ladder", "vqe"),
    )

    for items in paths:
        fpath = items[0]

        alg = items[2]

        ledm = ExperimentDataManager.load(Path(DATA_DIR, "1qp", fpath))
        d = ledm.load_last_saved_data_file
        J = d["inputs"]["J"]
        true_disp = d["results"]["ed"]["disp"]["statevector"]
        disp = d["results"][alg]["disp"]["real_backend"]

        handles[fpath] = ax.errorbar(
            J,
            np.mean(np.abs(np.array(true_disp) - np.array(disp))),
            # yerr=np.mean(np.abs(np.array(true_disp) - np.array(disp))),
            fmt=ALGSYM[alg],
            color=getcolor(fpath),
            label=fpath,
        )

    hd = [
        handles["vqe_tfim_30j_0hl_5q"],
        handles["vqe_tfim_50j_10hl_5q"],
        handles["sweep_tfim_30j_0hl_5q"],
        handles["vqe_tfim_20j_0hl_8q_ladder"],
    ]
    lbs = ["VQE", "VQE LF", "ASP", "VQE ladder"]

    ax.legend(
        hd,
        lbs,
        fontsize="x-small",
    )

    ax.set_xlabel(r"$J/h$")
    ax.set_ylabel(r"Mean dispersion error")

    edm.save_dict({"paths": paths})

    edm.save_figure(fig, "all_real_backend_errors", False, fig_shape="regular")

    plt.show()
