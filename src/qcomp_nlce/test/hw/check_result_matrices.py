from data_manager import ExperimentDataManager
import numpy as np
from qcomp_nlce.utils.workflow_1d import heff_backend_type
from qcomp_nlce.utils.results import plot_disps, average_result_matrices, show_diffs
import matplotlib.pyplot as plt


def replace_eld():
    fpaths_aqt = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_11\workflow_tfim_hadam_estimator_AQT_200shots",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_estimator_AQT_200shots",
    )
    fpath_qexa = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_19\vqe_workflow_tfim_hadam_stefan_samples_QEXA_200000shots"

    edm = ExperimentDataManager.load(fpath_qexa)
    dqexa = edm.load_last_saved_data_file
    edm = ExperimentDataManager.load(fpaths_aqt[0])
    daqt2 = edm.load_last_saved_data_file
    edm = ExperimentDataManager.load(fpaths_aqt[1])
    daqt = edm.load_last_saved_data_file

    out_d = average_result_matrices([daqt, daqt2])

    for si in ("dict_l", "dict_s"):
        out_d["results"]["vqe"][si]["eld"] = dqexa["results"]["vqe"][si]["eld"]

    plot_disps(out_d)


def analyze_fpath():

    fpath_aqt = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_estimator_AQT_200shots_00001"
    fpath_qexa = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_19\vqe_workflow_tfim_hadam_stefan_samples_QEXA_200000shots"

    edm = ExperimentDataManager.load(fpath_qexa)
    d = edm.load_last_saved_data_file

    show_diffs(d)
    plot_disps(d)


if __name__ == "__main__":
    replace_eld()
