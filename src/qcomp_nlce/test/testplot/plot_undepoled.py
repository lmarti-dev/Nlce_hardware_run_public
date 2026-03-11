from qcomp_nlce.utils.results import undepolarize_out_dict
from data_manager import ExperimentDataManager
from qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)
import matplotlib.pyplot as plt
from fauplotstyle import style
from qcomp_nlce.utils import DATA_DIR
from pathlib import Path

style()

fpath_tfim_100j_0hl = r"2026_02_02\vqe_workflow_TFIM_hadam_stefan_AQT_200shots_00001"
fpath_tfim_80j_10hl = r"2026_02_06\vqe_workflow_TFIM_hl_10_hadam_stefan_AQT_200shots"
fpath_tfim_80j_0hl = (
    r"2026_03_03/vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_00001"
)

DEPOL_RATE = 0.0527

dry_run = False
m_edm = ExperimentDataManager(
    "plot_result_depol_errmit", project="nlce", dry_run=dry_run
)

for items in (
    (fpath_tfim_100j_0hl, "TFIM_100j_0hl"),
    (fpath_tfim_80j_10hl, "TFIM_80j_10hl"),
    (fpath_tfim_80j_0hl, "TFIM_80j_0hl"),
):
    fpath, fig_name = items
    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file
    if "model_dir" in d["inputs"]:
        which_model = d["inputs"]["model_dir"]
    else:
        which_model = d["inputs"]["which_model"]

    fig1 = plot_all_backends_dispersions_montecarlo(
        d, which_circ=["vqe", "ed"], which_model=fig_name
    )

    fig2 = plot_all_backends_dispersions_montecarlo(
        undepolarize_out_dict(d, DEPOL_RATE),
        which_circ=["vqe", "ed"],
        which_model=fig_name,
    )

    m_edm.var_dump(fpath=fpath)
    m_edm.save_figure(
        fig1,
        filename=f"{fig_name}_results_no_errmit",
        fig_shape="nice-y",
        add_timestamp=False,
    )
    m_edm.save_figure(
        fig2,
        filename=f"{fig_name}_results_with_errmit",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    # plt.show()
