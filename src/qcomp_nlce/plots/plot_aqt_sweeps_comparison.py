from fauplotstyle import style
import matplotlib.pyplot as plt
from src.qcomp_nlce.utils import (
    exact_dispersion_1qp_1d,
    BACKEND_LABELS,
    BACKEND_COLORS,
    DATA_DIR,
)
from data_manager import ExperimentDataManager
from pathlib import Path

style()

runs_current = (
    r"error_decrease/current/10steps",
    r"error_decrease/current/20steps",
)


runs_10improvement = (
    r"error_decrease/better10x/10steps",
    r"error_decrease/better10x/20steps",
)

runs_100improvement = (
    r"error_decrease/better100x/10steps",
    r"error_decrease/better100x/20steps",
)

dry_run = False
m_edm = ExperimentDataManager(
    f"plot_sweep_comparison_aqt", project="nlce", dry_run=dry_run
)

for crun in (
    ("tfim_100j_0hl_current", runs_current),
    ("tfim_100j_0hl_10x_better", runs_10improvement),
    ("tfim_100j_0hl_100x_better", runs_100improvement),
):
    name = crun[0]
    fpaths = crun[1]

    dicts = []
    for r in fpaths:
        edm = ExperimentDataManager.load(Path(DATA_DIR, r))
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

    ax.errorbar(
        ks,
        sv10,
        sverr10,
        capsize=2,
        fmt="x-",
        color=BACKEND_COLORS["statevector"],
        label=f"{sv_label}, 10 steps",
    )
    ax.errorbar(
        ks,
        sv20,
        sverr20,
        capsize=2,
        fmt="o-",
        color=BACKEND_COLORS["statevector"],
        label=f"{sv_label}, 20 steps",
    )
    ax.errorbar(
        ks,
        fb10,
        fberr10,
        capsize=2,
        fmt="x-",
        color=BACKEND_COLORS["fake_backend"],
        label=f"{fb_label}, 10 steps",
    )
    ax.errorbar(
        ks,
        fb20,
        fberr20,
        capsize=2,
        fmt="o-",
        color=BACKEND_COLORS["fake_backend"],
        label=f"{fb_label}, 20 steps",
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
        fig, filename=f"sweep_aqt_{name}", fig_shape="nice-y", add_timestamp=False
    )
    # plt.show()
