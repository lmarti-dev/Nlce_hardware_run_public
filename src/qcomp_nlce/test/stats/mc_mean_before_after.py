from fauplotstyle import style
import matplotlib.pyplot as plt
from src.qcomp_nlce.utils import (
    mc_disp_from_res_dict,
    BACKEND_LABELS,
    BACKEND_COLORS,
    DATA_DIR,
)
from data_manager import ExperimentDataManager
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    style()

    fpath = Path(DATA_DIR, "1qp/vqe_tfim_80j_0hl_5q")

    dry_run = False
    m_edm = ExperimentDataManager(
        "plot_mc_mean_before_after", project="nlce", dry_run=dry_run
    )

    backend = "statevector"
    fig, ax = plt.subplots()
    edm = ExperimentDataManager.load(fpath)
    d = edm.load_last_saved_data_file

    ks = d["inputs"]["ks"]
    j = d["inputs"]["J"]

    shots = d["inputs"]["n_reps"] * d["inputs"]["n_shots"]
    name = d["inputs"]["model_dir"]

    ed_disp = d["results"]["ed"]["disp"]["statevector"]

    label = BACKEND_LABELS[backend]

    n_samples = 10000

    print("\n")
    print(f"shots: {shots}")
    disp_before = mc_disp_from_res_dict(
        ks,
        d,
        "vqe",
        n_samples=n_samples,
        to_eotm=True,
    )

    disp_after = mc_disp_from_res_dict(
        ks,
        d,
        "vqe",
        n_samples=n_samples,
        to_eotm=False,
    )

    ax.errorbar(
        ks,
        disp_after["fake_backend"]["mean"],
        disp_after["fake_backend"]["std"] / np.sqrt(shots),
        capsize=2,
        fmt="x-",
        linewidth=1,
        label="After",
    )

    ax.errorbar(
        ks,
        disp_before["fake_backend"]["mean"],
        disp_before["fake_backend"]["std"],
        capsize=2,
        fmt="x-",
        linewidth=1,
        label="Before",
    )

    ax.plot(ks, ed_disp, label="ED")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega(k)$")

    ax.legend()

    m_edm.save_figure(
        fig,
        filename=f"mc_std_{backend}_{name}",
        fig_shape="nice-y",
        add_timestamp=False,
    )
    plt.show()
