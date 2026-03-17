from data_manager import ExperimentDataManager
import matplotlib.pyplot as plt
from fauplotstyle import style
from qcomp_nlce.utils import DATA_DIR
from pathlib import Path
import numpy as np

style()

fpath_tfim_100j_0hl = r"vqe_tfim_100j_0hl_5q"
fpath_tfim_80j_0hl_sweep = r"sweep_tfim_80j_0hl_5q"
fpath_tfim_80j_0hl = r"vqe_tfim_80j_0hl_5q"
fpath_tfim_80j_0hl_run2 = r"vqe_tfim_80j_0hl_5q_run2"
fpath_tfim_50j_10hl = r"vqe_tfim_50j_10hl_5q"


for items in (
    (fpath_tfim_80j_0hl, "TFIM_80j_0hl", "vqe"),
    (fpath_tfim_80j_0hl_run2, "TFIM_80j_0hl", "vqe"),
    (fpath_tfim_50j_10hl, "TFIM_50j_10hl", "vqe"),
    (fpath_tfim_80j_0hl_sweep, "TFIM_80j_0hl", "sweep"),
    (fpath_tfim_100j_0hl, "TFIM_100j_0hl", "vqe"),
):
    fpath, fig_name, alg = items
    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    edm.use_calendar = False
    d = edm.load_last_saved_data_file
    if "model_dir" in d["inputs"]:
        which_model = d["inputs"]["model_dir"]
    else:
        which_model = d["inputs"]["which_model"]

    print(f"{fig_name}")
    for be in ("fake_backend", "real_backend", "statevector"):
        for size in ("dict_l", "dict_s"):
            eld_std = np.abs(np.mean(d["results"][alg][size]["eld"][be]["stds"]))
            exd_std = np.abs(np.mean(d["results"][alg][size]["exd"][be]["stds"]))
            gsd_std = d["results"][alg][size]["gsd"][be]["std"]

            print(
                f"({size}, {be}) exd std {exd_std:.4e}, eld std {eld_std:.4e} gsd std {gsd_std:.4e}"
            )
        mc_std = np.abs(np.mean(d["results"][alg]["mc_disp"][be]["std"]))
        print(f"mc std: {mc_std:.4e}")
    # plt.show()
