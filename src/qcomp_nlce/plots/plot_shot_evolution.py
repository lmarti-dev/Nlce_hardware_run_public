from fauplotstyle import style
import matplotlib.pyplot as plt
from qcomp_nlce.utils import (
    exact_dispersion_1qp_1d,
    BACKEND_LABELS,
    BACKEND_COLORS,
    DATA_DIR,
)
from data_manager import ExperimentDataManager
from pathlib import Path

style()

fpaths = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_11\vqe_workflow_vqe_100j_0hl_hadam_stefan_AQT_200shots_x_2reps",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_11\vqe_workflow_vqe_100j_0hl_hadam_stefan_AQT_200shots_x_10reps",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_11\vqe_workflow_vqe_100j_0hl_hadam_stefan_AQT_200shots_x_40reps",
)


dry_run = False
m_edm = ExperimentDataManager(
    f"plot_n_shots_effect_comp", project="nlce", dry_run=dry_run
)


for backend in ("fake_backend", "statevector"):
    fig, ax = plt.subplots()
    for fpath in fpaths:
        edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
        d = edm.load_last_saved_data_file

        ks = d["inputs"]["ks"]
        j = d["inputs"]["J"]

        shots = d["inputs"]["n_reps"] * d["inputs"]["n_shots"]
        name = d["inputs"]["model_dir"]

        disp = d["results"]["vqe"]["disp"][backend]
        dis_std = d["results"]["vqe"]["mc_disp"][backend]["std"]
        ed = d["results"]["ed"]["disp"]["statevector"]
        ana = exact_dispersion_1qp_1d(j, ks)

        label = BACKEND_LABELS[backend]

        ax.errorbar(
            ks,
            disp,
            dis_std,
            capsize=2,
            fmt="o-",
            linewidth=0.25,
            label=f"{label}, {shots}",
        )

    ax.plot(ks, ed, label="ED", color=BACKEND_COLORS["ed"])
    if "_0hl" in name:
        ax.plot(ks, ana, label="Analytic", color=BACKEND_COLORS["analytic"])

    ax.set_ylabel(r"$\omega(k)$")
    ax.set_xlabel("$k$")
    # ax.set_yscale("log")
    ax.legend(
        ncol=2,
        fontsize="x-small",
        columnspacing=1,
    )
    m_edm.save_figure(
        fig,
        filename=f"shot_comp_{backend}_vqe_{name}",
        fig_shape="nice-y",
        add_timestamp=False,
    )
    # plt.show()
