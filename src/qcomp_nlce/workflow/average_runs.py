from data_manager import ExperimentDataManager
from qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)

from qcomp_nlce.utils.results import average_result_matrices, show_diffs

import matplotlib.pyplot as plt

from data_manager import ExperimentDataManager
from pathlib import Path


def load_diffs(fpath: Path):
    edm = ExperimentDataManager.load(fpath)
    d = edm.load_last_saved_data_file
    show_diffs(d, "real_backend")


def plot_average_over_runs(dry_run: bool = True):
    runs = (
        r"C:/Users/Moi4/Desktop/current/FAU/phd/data\2026_01_28\vqe_workflow_tfim_hadam_stefan_AQT_200shots_00001",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_11\workflow_tfim_hadam_estimator_AQT_200shots",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_estimator_AQT_200shots",
    )

    m_edm = ExperimentDataManager("plot_average_runs", project="nlce", dry_run=dry_run)

    dicts = []
    for r in runs:
        edm = ExperimentDataManager.load(r)
        d = edm.load_last_saved_data_file
        dicts.append(d)

    to_average = ["exd", "eld", "gsd"]
    out_d = average_result_matrices(
        dicts, to_average=to_average, backends=["real_backend", "statevector", "system"]
    )

    fig = plot_all_backends_dispersions_montecarlo(out_d)

    m_edm.var_dump(files=runs, to_average=to_average, ref_dict=runs[0])
    m_edm.save_figure(fig, fig_shape="double-y")

    plt.show()


if __name__ == "__main__":

    # fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00005"
    # plot_average_over_runs()
    runs = (
        r"C:/Users/Moi4/Desktop/current/FAU/phd/data\2026_01_28\vqe_workflow_tfim_hadam_stefan_AQT_200shots_00001",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_11\workflow_tfim_hadam_estimator_AQT_200shots",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_estimator_AQT_200shots",
    )
    for r in runs:
        print(r)
        load_diffs(r)
