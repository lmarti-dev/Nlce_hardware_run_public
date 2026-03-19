from data_manager import ExperimentDataManager
from src.qcomp_nlce.utils.workflow_utils import (
    plot_0qp_gs_nlce,
    plot_bitstring_props,
)
from src.qcomp_nlce.utils import DATA_DIR
import matplotlib.pyplot as plt
from fauplotstyle import style
from pathlib import Path

style()

fpath_tfim_80j_0hl = r"vqe_0qp_tfim_7q_8q"

dry_run = False
m_edm = ExperimentDataManager(
    "plot_0qp_tfim_qpu_result", project="nlce", dry_run=dry_run
)

for items in ((fpath_tfim_80j_0hl, "TFIM_80j_0hl"),):
    fpath, fig_name = items
    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file

    fig = plot_0qp_gs_nlce(d)

    m_edm.var_dump(fpath=fpath)
    m_edm.save_figure(
        fig,
        filename=f"gs_0qp_{fig_name}_results",
        fig_shape="nice-y",
        add_timestamp=False,
    )
    plt.show()

    fig2 = plot_bitstring_props(d)

    m_edm.save_figure(
        fig2,
        filename=f"gs_0qp_{fig_name}_results_bitprop",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    plt.show()
