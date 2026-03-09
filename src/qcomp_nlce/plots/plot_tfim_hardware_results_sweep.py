from data_manager import ExperimentDataManager
from qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)
import matplotlib.pyplot as plt
from fauplotstyle import style
from qcomp_nlce.utils import DATA_DIR
from pathlib import Path

style()


fpath_tfim_80j_0hl = r"2026_03_07/sweep_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots"


dry_run = False
m_edm = ExperimentDataManager(
    "plot_tfim_sweep_qpu_result", project="nlce", dry_run=dry_run
)

for items in ((fpath_tfim_80j_0hl, "TFIM_80j_0hl"),):
    fpath, fig_name = items
    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file
    if "model_dir" in d["inputs"]:
        which_model = d["inputs"]["model_dir"]
    else:
        which_model = d["inputs"]["which_model"]

    fig = plot_all_backends_dispersions_montecarlo(
        d, which_circ=["sweep", "ed"], which_model=fig_name
    )

    m_edm.var_dump(fpath=fpath)
    m_edm.save_figure(
        fig,
        filename=f"sweep_{fig_name}_results",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    plt.show()
