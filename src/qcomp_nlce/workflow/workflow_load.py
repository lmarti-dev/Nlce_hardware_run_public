import copy
from data_manager import ExperimentDataManager
from Nlce_hardware_run.src.qcomp_nlce.utils.workflow_tfim_1d import (
    heff_backend_type,
    dispersion_all_backends,
    plot_all_backends_dispersions_montecarlo,
)

from qcomp_nlce.utils.results import average_result_matrices

import matplotlib.pyplot as plt
import numpy as np

from data_manager import ExperimentDataManager


def plot_average_over_runs(dry_run: True):
    runs = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_11\workflow_tfim_hadam_estimator_AQT_200shots",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_estimator_AQT_200shots",
        r"C:/Users/Moi4/Desktop/current/FAU/phd/data\2026_01_28\vqe_workflow_tfim_hadam_stefan_AQT_200shots_00001",
    )

    m_edm = ExperimentDataManager("plot_average_runs", project="nlce")

    dicts = []
    for r in runs:
        edm = ExperimentDataManager.load(r)
        d = edm.load_last_saved_data_file
        dicts.append(d)

    out_d = average_result_matrices(dicts, to_average=["eld", "gsd"])

    fig = plot_all_backends_dispersions_montecarlo(out_d)

    plt.show()


if __name__ == "__main__":

    fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00005"
    plot_average_over_runs()
