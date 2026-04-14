from numpy import shape
from data_manager import ExperimentDataManager
from src.qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)
import matplotlib.pyplot as plt
from fauplotstyle import style
from src.qcomp_nlce.utils import DATA_DIR
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm

def plot_offdiag_elems(dry_run):
    style()

    m_edm = ExperimentDataManager(
        "plot_offdiag_sv", project="nlce", dry_run=dry_run
    )

    nrows=2
    ncols=5

    fig,axes = plt.subplots(figsize=(8,1),nrows=nrows,ncols=ncols,constrained_layout=True)

    for jj,items in enumerate((
        ("vqe_tfim_30j_0hl_5q", "TFIM_30j_0hl", "vqe"),
        ("vqe_tfim_80j_0hl_5q", "TFIM_80j_0hl", "vqe"),
        ("vqe_tfim_100j_0hl_5q", "TFIM_100j_0hl", "vqe"),
        ("vqe_tfim_50j_10hl_5q", "TFIM_50j_10hl", "vqe"),
        ("vqe_tfim_20j_0hl_8q_ladder", "TFIM_20j_0hl_ladder", "vqe"),)
    ):
        fpath, fig_name, alg = items
        edm = ExperimentDataManager.load(Path(DATA_DIR, "1qp", fpath))
        edm.use_calendar = False
        d = edm.load_last_saved_data_file
        if "model_dir" in d["inputs"]:
            which_model = d["inputs"]["model_dir"]
        else:
            which_model = d["inputs"]["which_model"]

        J = d["inputs"]["J"]
        hl = d["inputs"]["hl"]
        q = d["inputs"]["n_qubits"]
        ladder = d["inputs"]["ladder"]


        for kk, size in enumerate(("dict_l","dict_s")):
            qstr = q
            if size=="dict_s":
                qstr = q-1
            if ladder:
                qstr = fr"{q} \times 2"
            if hl==0:
                hstr = ""
            else:
                hstr = fr", h_l={hl:.1f}"
            mat = d["results"][alg][size]["eld"]["statevector"]["values"]
            ax:plt.Axes = axes[kk,jj]
            ax.imshow(np.abs(mat)/np.max(np.abs(mat)),norm=LogNorm(vmin=0.01, vmax=1))
            ax.set_title(fr"$J/h={J:.1f}{hstr}, {qstr}q$")
            ax.set_xticks([])
            ax.set_yticks([])
        

    m_edm.var_dump(fpath=fpath)
    m_edm.save_figure(
        fig,
        filename="offdiag_elems_sv",    
        fig_shape="page-wide",
        add_timestamp=False,
    )

    plt.show()
