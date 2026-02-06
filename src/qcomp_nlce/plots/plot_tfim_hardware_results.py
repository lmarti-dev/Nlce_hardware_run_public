from data_manager import ExperimentDataManager
from qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)
import matplotlib.pyplot as plt
from fauplotstyle import style


style()

dry_run = False
m_edm = ExperimentDataManager("plot_tfim_qpu_result", project="nlce", dry_run=dry_run)
fpath_tfim = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_02\vqe_workflow_TFIM_hadam_stefan_AQT_200shots_00001"
fpath_tfim_hl = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_06\vqe_workflow_TFIM_hl_10_hadam_stefan_AQT_200shots"

for fpath in (fpath_tfim, fpath_tfim_hl):

    edm = ExperimentDataManager.load(fpath)
    d = edm.load_last_saved_data_file

    which_model = d["inputs"]["which_model"]
    fig = plot_all_backends_dispersions_montecarlo(d)

    m_edm.var_dump(fpath=fpath)
    m_edm.save_figure(
        fig,
        filename=f"{which_model}_results",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    plt.show()
