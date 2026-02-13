from fauplotstyle import style
import matplotlib.pyplot as plt
from qcomp_nlce.utils import exact_dispersion_1qp_1d, BACKEND_LABELS
from data_manager import ExperimentDataManager

style()

runs_current = (
    r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_29\vqe_workflow_TFIM_hadam_stefan_AQT_2000shots_00001",
    r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_29\vqe_workflow_TFIM_hadam_stefan_AQT_2000shots",
)


runs_10improvement = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_30\vqe_workflow_TFIM_hadam_stefan_AQT_BETTER_10_2000shots",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_30\vqe_workflow_TFIM_hadam_stefan_AQT_BETTER_10_2000shots_00001",
)

runs_100improvement = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_30\vqe_workflow_TFIM_hadam_stefan_AQT_BETTER_100_2000shots",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_30\vqe_workflow_TFIM_hadam_stefan_AQT_BETTER_100_2000shots_00001",
)


runs_80j_0hl = (
    r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_12\vqe_workflow_TFIM_hadam_stefan_AQT_2000shots_00001",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_12\vqe_workflow_TFIM_hadam_stefan_AQT_2000shots_00002",
)

runs_80j_10hl = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_80j_10hl_hadam_stefan_AQT_2000shots_00001",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_80j_10hl_hadam_stefan_AQT_2000shots_00002",
)

runs_80j_50hl = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_80j_50hl_hadam_stefan_AQT_2000shots",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_80j_50hl_hadam_stefan_AQT_2000shots_00001",
)

runs_80j_100hl = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_80j_100hl_hadam_stefan_AQT_2000shots",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_80j_100hl_hadam_stefan_AQT_2000shots",
)

runs_100j_10hl = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_100j_10hl_hadam_stefan_AQT_2000shots",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_100j_10hl_hadam_stefan_AQT_2000shots_00001",
)

runs_100j_50hl = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_100j_50hl_hadam_stefan_AQT_2000shots",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_100j_50hl_hadam_stefan_AQT_2000shots_00001",
)

runs_100j_100hl = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_100j_100hl_hadam_stefan_AQT_2000shots",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_13\sweep_workflow_vqe_100j_100hl_hadam_stefan_AQT_2000shots_00001",
)


dry_run = False
m_edm = ExperimentDataManager(
    f"plot_sweep_comparison_aqt", project="nlce", dry_run=dry_run
)

for crun in (
    ("tfim_100j_0hl_current", runs_current),
    ("tfim_100j_0hl_10x_better", runs_10improvement),
    ("tfim_100j_0hl_100x_better", runs_100improvement),
    ("tfim_80j_0hl", runs_80j_0hl),
    ("tfim_80j_10hl", runs_80j_10hl),
    ("tfim_80j_50hl", runs_80j_50hl),
    ("tfim_80j_100hl", runs_80j_100hl),
    ("tfim_100j_10hl", runs_100j_10hl),
    ("tfim_100j_50hl", runs_100j_50hl),
    ("tfim_100j_100hl", runs_100j_100hl),
):
    name = crun[0]
    fpaths = crun[1]

    dicts = []
    for r in fpaths:
        edm = ExperimentDataManager.load(r)
        d = edm.load_last_saved_data_file
        dicts.append(d)

    d10, d20 = dicts
    fig, ax = plt.subplots()

    ks = d10["inputs"]["ks"]
    j = d10["inputs"]["J"]

    sv10 = d10["results"]["sweep"]["disp"]["statevector"]
    sverr10 = d10["results"]["sweep"]["mc_disp"]["statevector"]["std"]
    sv20 = d20["results"]["sweep"]["disp"]["statevector"]
    sverr20 = d20["results"]["sweep"]["mc_disp"]["statevector"]["std"]
    fb10 = d10["results"]["sweep"]["disp"]["fake_backend"]
    fberr10 = d10["results"]["sweep"]["mc_disp"]["fake_backend"]["std"]
    fb20 = d20["results"]["sweep"]["disp"]["fake_backend"]
    fberr20 = d20["results"]["sweep"]["mc_disp"]["fake_backend"]["std"]
    ed = d10["results"]["ed"]["disp"]["statevector"]
    ana = exact_dispersion_1qp_1d(j, ks)

    sv_label = BACKEND_LABELS["statevector"]
    fb_label = BACKEND_LABELS["fake_backend"]

    ax.errorbar(ks, sv10, sverr10, capsize=2, fmt="x-", label=f"{sv_label}, 10 steps")
    ax.errorbar(ks, sv20, sverr20, capsize=2, fmt="x-", label=f"{sv_label}, 20 steps")
    ax.errorbar(ks, fb10, fberr10, capsize=2, fmt="o-", label=f"{fb_label}, 10 steps")
    ax.errorbar(ks, fb20, fberr20, capsize=2, fmt="o-", label=f"{fb_label}, 20 steps")
    ax.plot(ks, ed, "+-", label="ED")
    if "_0hl" in name:
        ax.plot(ks, ana, "d-", label="Analytic")

    ax.set_ylabel(r"$\omega(k)$")
    ax.set_xlabel("$k$")
    # ax.set_yscale("log")
    ax.legend(
        ncol=2,
        fontsize="x-small",
        columnspacing=1,
    )
    m_edm.save_figure(
        fig, filename=f"sweep_aqt_{name}", fig_shape="nice-y", add_timestamp=False
    )
    plt.show()
