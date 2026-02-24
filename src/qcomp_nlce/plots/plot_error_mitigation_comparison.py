from fauplotstyle import style
import matplotlib.pyplot as plt
from qcomp_nlce.utils import exact_dispersion_1qp_1d, BACKEND_LABELS
from data_manager import ExperimentDataManager
import numpy as np


def ldif(l1: list, l2: list) -> np.ndarray:
    return np.abs(np.array(l1) - np.array(l2))


style()

with_mit = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_23\vqe_workflow_vqe_80j_0hl_hadam_stefan_samples_EQE_1000000shots"
without_mit = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_23\vqe_workflow_vqe_80j_0hl_hadam_stefan_samples_EQE_1000000shots_00001"

dry_run = False
m_edm = ExperimentDataManager(f"plot_error_mit_comp", project="nlce", dry_run=dry_run)

for crun in (("tfim_80j_0hl_current", (without_mit, with_mit)),):
    name = crun[0]
    fpaths = crun[1]

    dicts = []
    for r in fpaths:
        edm = ExperimentDataManager.load(r)
        d = edm.load_last_saved_data_file
        dicts.append(d)

    with_mit, without_mit = dicts
    fig, ax = plt.subplots()

    ks = with_mit["inputs"]["ks"]
    j = with_mit["inputs"]["J"]

    sv_with = with_mit["results"]["vqe"]["disp"]["statevector"]
    sverr_with = with_mit["results"]["vqe"]["mc_disp"]["statevector"]["std"]
    sv_without = without_mit["results"]["vqe"]["disp"]["statevector"]
    sverr_without = without_mit["results"]["vqe"]["mc_disp"]["statevector"]["std"]
    fb_with = with_mit["results"]["vqe"]["disp"]["fake_backend"]
    fberr_with = with_mit["results"]["vqe"]["mc_disp"]["fake_backend"]["std"]
    fb_without = without_mit["results"]["vqe"]["disp"]["fake_backend"]
    fberr_without = without_mit["results"]["vqe"]["mc_disp"]["fake_backend"]["std"]
    ed = with_mit["results"]["ed"]["disp"]["statevector"]
    ana = exact_dispersion_1qp_1d(j, ks)

    sv_label = BACKEND_LABELS["statevector"]
    fb_label = BACKEND_LABELS["fake_backend"]

    # ax.errorbar(
    #     ks,
    #     sv_with,
    #     sverr_with,
    #     capsize=2,
    #     fmt="x-",
    #     label=f"{sv_label}, w/ EM",
    # )
    # ax.errorbar(
    #     ks,
    #     sv_without,
    #     sverr_without,
    #     capsize=2,
    #     fmt="x-",
    #     label=f"{sv_label}, w/o EM",
    # )

    spread = np.abs(max(ana) - min(ana))

    ax.errorbar(
        ks,
        ldif(fb_with, sv_with) / spread,
        fberr_with,
        capsize=2,
        fmt="o-",
        label=f"{fb_label}, w/ EM",
    )
    ax.errorbar(
        ks,
        ldif(fb_without, sv_without) / spread,
        fberr_without,
        capsize=2,
        fmt="o-",
        label=f"{fb_label}, w/o EM",
    )
    # ax.plot(ks, ed, "+-", label="ED")
    # if "_0hl" in name:
    #     ax.plot(ks, ana, "d-", label="Analytic")

    ax.set_ylabel(r"$|\tilde{\omega}(k)-\omega(k)|/\Delta\omega$")
    ax.set_xlabel("$k$")
    # ax.set_yscale("log")
    ax.legend(
        ncol=2,
        fontsize="x-small",
        columnspacing=1,
    )
    m_edm.save_figure(
        fig, filename=f"em_comp_{name}", fig_shape="nice-y", add_timestamp=False
    )
    plt.show()
