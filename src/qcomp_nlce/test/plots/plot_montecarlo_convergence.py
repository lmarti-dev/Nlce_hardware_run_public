from data_manager import ExperimentDataManager
from qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
    mc_disp_from_res_dict,
)
import matplotlib.pyplot as plt
from fauplotstyle import style


dry_run = True
m_edm = ExperimentDataManager("plot_tfim_qpu_result", project="nlce", dry_run=dry_run)
fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_02\vqe_workflow_TFIM_hadam_stefan_AQT_200shots"


edm = ExperimentDataManager.load(fpath)
d = edm.load_last_saved_data_file


mc_disp = mc_disp_from_res_dict(ks, out_dict, "vqe", n_mc_samples, to_eotm=True)
out_dict["results"]["vqe"]["mc_disp"] = mc_disp
