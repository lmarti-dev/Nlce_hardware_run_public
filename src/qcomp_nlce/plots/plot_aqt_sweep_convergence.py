from fauplotstyle import style
import matplotlib.pyplot as plt
from src.qcomp_nlce.utils import (
    DATA_DIR,
    exact_dispersion_1qp_1d,
)
from data_manager import ExperimentDataManager
from pathlib import Path
import numpy as np


def plot_sweep_convergence(fpaths):
    dry_run = False
    m_edm = ExperimentDataManager(
        "plot_sweep_convergence", project="nlce", dry_run=dry_run
    )

    edm = ExperimentDataManager.load(Path(DATA_DIR, fpaths[0]))
    d = edm.load_last_saved_data_file
    ks = d["inputs"]["ks"]
    name = d["inputs"]["model_dir"]
    j = d["inputs"]["J"]
    true_disp = exact_dispersion_1qp_1d(j, ks)

    steps = []
    errs = []
    stds = []

    for fpath in fpaths:
        edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
        d = edm.load_last_saved_data_file
        sv_disp = d["results"]["sweep"]["disp"]["statevector"]
        n_steps = d["inputs"]["n_steps"]

        steps.append(n_steps)
        errs.append(np.mean(np.abs(sv_disp - true_disp)))
        stds.append(np.std(np.abs(sv_disp - true_disp)))

    steps = np.array(steps)
    errs = np.array(errs)
    stds = np.array(stds)

    iij = np.argsort(steps)

    fig, ax = plt.subplots()
    ax.errorbar(
        steps[iij],
        errs[iij],
        stds[iij],
        capsize=2,
        fmt="x:",
        linewidth=1,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of sweep steps")
    ax.set_ylabel("Mean error. of dispersion")

    m_edm.save_figure(
        fig,
        filename="sweep_conv_tfim_80j",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    plt.show()


if __name__ == "__main__":
    style()

    fpaths = [
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00011",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00001",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00002",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00003",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00004",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00005",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00006",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00007",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00008",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00009",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_23\sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00010",
    ]

    plot_sweep_convergence(fpaths)
