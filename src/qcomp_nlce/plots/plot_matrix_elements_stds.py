from data_manager import ExtendedJSONDecoder

import io
import json

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal
from qcomp_nlce.utils import BACKEND_TYPES, DATA_DIR
from data_manager import ExperimentDataManager
from fauplotstyle import style


def load_json(fpath: Path) -> list:
    with io.open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)
    return jobj


def get_matrix(
    jobj: dict,
    size: Literal["dict_l", "dict_s"],
    which_matrix: Literal["exs", "eld"],
    which_backend: BACKEND_TYPES,
):
    return (
        jobj["results"]["vqe"][size][which_matrix][which_backend]["values"],
        jobj["results"]["vqe"][size][which_matrix][which_backend]["stds"],
    )


def get_gs(
    jobj: dict,
    size: Literal["dict_l", "dict_s"],
    which_backend: BACKEND_TYPES,
):
    return (
        jobj["results"]["vqe"][size]["gsd"][which_backend]["energy"],
        jobj["results"]["vqe"][size]["gsd"][which_backend]["std"],
    )


if __name__ == "__main__":

    style()
    dry_run = False
    which_backend = "real_backend"

    edm = ExperimentDataManager(
        f"plot_meshes_{which_backend}", project="nlce", dry_run=dry_run
    )
    aqt_fpath = r"2026_02_02\vqe_workflow_TFIM_hadam_stefan_AQT_200shots_00001\run_00000\data\macadamias_00000.json"

    jobj = load_json(Path(DATA_DIR, aqt_fpath))
    if "model_dir" in jobj["inputs"]:
        which_model = jobj["inputs"]["model_dir"]
    else:
        which_model = jobj["inputs"]["which_model"]

    textcolors = ["k", "w"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), constrained_layout=True)
    pn = jobj["inputs"]["provider_name"]
    nq = jobj["inputs"]["n_qubits"]
    n_shots = jobj["inputs"]["n_shots"]
    n_reps = jobj["inputs"]["n_reps"]
    n_shots *= n_reps

    SIZES_D = {"dict_l": f"{nq} qubits", "dict_s": f"{nq-1} qubits"}
    WHICH_D = {
        "eld": rf"$\langle j | U | k \rangle$",
        "exd": rf"$\langle j | U^\dagger H U | k \rangle$",
    }

    edm.var_dump(**jobj["inputs"], aqt_fpath=aqt_fpath)

    for ind_x, size in enumerate(("dict_l", "dict_s")):
        for ind_y, which in enumerate(("eld", "exd")):
            mat, stds = get_matrix(jobj, size, which, which_backend)
            stds /= np.sqrt(n_shots)
            nq = len(mat)
            astds = np.abs(stds)
            ax: plt.Axes = axes[ind_x, ind_y]
            im = ax.imshow(np.abs(stds))
            threshold = im.norm(np.max(astds)) / 2

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ttype = "val"
                    if ttype == "val":
                        sign = "" if mat[i, j].imag >= 0 else "-"
                        s = f"{mat[i, j].real:.3f}\n{sign}{np.abs(mat[i, j].imag):.3f}"
                    elif ttype == "std":
                        rstd = np.abs(stds[i, j].real)
                        istd = np.abs(stds[i, j].imag)
                        s = f"{float(rstd):.2f}\n{float(istd):.2f}"
                    text = ax.text(
                        j,
                        i,
                        s,
                        ha="center",
                        va="center",
                        color=textcolors[int(im.norm(astds[i, j]) < threshold)],
                    )
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(rf"{SIZES_D[size]} - {WHICH_D[which]}")
            ax.grid(False)

        ax.set_xticks(range(nq), range(1, 1 + nq))
        ax.set_yticks(range(nq), range(1, 1 + nq))
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0.01, wspace=0.5)
    edm.save_figure(
        fig,
        f"{which_backend}_{which_model}_meshes_std",
        fig_shape="double-size",
        add_timestamp=False,
    )
    plt.show()
