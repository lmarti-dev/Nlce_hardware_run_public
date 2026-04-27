from data_manager import ExperimentDataManager
from src.qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)
import matplotlib.pyplot as plt
from fauplotstyle import style
from src.qcomp_nlce.utils import DATA_DIR
from pathlib import Path

style()

dry_run = False
m_edm = ExperimentDataManager(
    "plot_tfim_vqe_qpu_result", project="nlce", dry_run=dry_run
)

for items in (
    ("vqe_tfim_30j_0hl_5q", "TFIM_30j_0hl", "vqe"),
    ("vqe_tfim_80j_0hl_5q", "TFIM_80j_0hl", "vqe"),
    ("vqe_tfim_80j_0hl_5q_run2", "TFIM_80j_0hl", "vqe"),
    ("vqe_tfim_50j_10hl_5q", "TFIM_50j_10hl", "vqe"),
    ("vqe_tfim_50j_10hl_5q_run2", "TFIM_50j_10hl", "vqe"),
    ("vqe_tfim_50j_10hl_5q_run3", "TFIM_50j_10hl", "vqe"),
    ("sweep_tfim_30j_0hl_5q", "TFIM_30j_0hl", "sweep"),
    ("sweep_tfim_80j_0hl_5q", "TFIM_80j_0hl", "sweep"),
    ("sweep_tfim_80j_0hl_5q_run2", "TFIM_80j_0hl", "sweep"),
    ("vqe_tfim_100j_0hl_5q", "TFIM_100j_0hl", "vqe"),
    ("vqe_tfim_100j_0hl_5q_run2", "TFIM_100j_0hl", "vqe"),
    ("vqe_tfim_100j_0hl_5q_run3", "TFIM_100j_0hl", "vqe"),
    ("vqe_tfim_20j_0hl_8q_ladder", "TFIM_20j_0hl_ladder", "vqe"),
):
    fpath, fig_name, alg = items
    edm = ExperimentDataManager.load(Path(DATA_DIR, "1qp", fpath))
    edm.use_calendar = False
    d = edm.load_last_saved_data_file
    if "model_dir" in d["inputs"]:
        which_model = d["inputs"]["model_dir"]
    else:
        which_model = d["inputs"]["which_model"]

    fig = plot_all_backends_dispersions_montecarlo(
        d,
        which_circ=[alg, "ed"],
        which_model=fig_name,
        which_backends=["real_backend", "statevector"],
    )

    m_edm.var_dump(fpath=fpath)
    m_edm.save_figure(
        fig,
        filename=f"{fig_name}_results_{alg}",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    # plt.show()
