from fauplotstyle import style
import matplotlib.pyplot as plt
from qcomp_nlce.utils import (
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

    fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_11\vqe_workflow_vqe_100j_0hl_hadam_stefan_AQT_200shots_x_40reps"

    dry_run = True
    m_edm = ExperimentDataManager(
        f"plot_mc_sanity_test", project="nlce", dry_run=dry_run
    )

    backend = "statevector"
    fig, ax = plt.subplots()
    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file

    ks = d["inputs"]["ks"]
    j = d["inputs"]["J"]

    shots = d["inputs"]["n_reps"] * d["inputs"]["n_shots"]
    name = d["inputs"]["model_dir"]

    label = BACKEND_LABELS[backend]

    shot_pows = (0, 1, 2, 3, 4, 5)
    n_samples = 100

    shot_list = []
    mc_stds = []
    std_on_std = []
    for pw in shot_pows:
        newrep = 10**pw
        d["inputs"]["n_reps"] = newrep
        shots = d["inputs"]["n_reps"] * d["inputs"]["n_shots"]
        print("\n")
        print(f"shots: {shots}")
        stds_in = 1 / np.sqrt(shots)
        disp = mc_disp_from_res_dict(
            ks,
            d,
            "vqe",
            n_samples=n_samples,
            to_eotm=True,
            gsd_sup_fac=1 / np.sqrt(newrep),
        )
        mc_stds.append(np.mean(disp[backend]["std"]))
        std_on_std.append(np.mean(disp[backend]["std"]))
        shot_list.append(shots)
    ax.errorbar(
        shot_list,
        mc_stds,
        std_on_std,
        capsize=2,
        fmt="x-",
        linewidth=1,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_{samples}$")
    ax.set_ylabel("Standard deviation")

    m_edm.save_figure(
        fig,
        filename=f"mc_std_{backend}_{name}",
        fig_shape="nice-y",
        add_timestamp=False,
    )
    plt.show()
