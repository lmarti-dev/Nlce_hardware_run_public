from data_manager import ExtendedJSONDecoder

import io
import re
import json

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal
from src.qcomp_nlce.utils import BACKEND_TYPES, DATA_DIR, ALGORITHMS_ALIAS, OFFDIAG_SYMS
from data_manager import ExperimentDataManager
from fauplotstyle import style


def load_json(fpath: Path) -> list:
    with io.open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)
    return jobj


def compute_depol_elem(noisy: float, ideal: float) -> float:
    if np.isclose(ideal, 0) or np.isclose(ideal, noisy):
        depol = 0.0
    elif noisy / ideal > 1 or noisy / ideal < 0:
        depol = np.nan
    else:
        depol = 1 - noisy / ideal

    return depol


def compute_depol(mat: np.ndarray, true_mat: np.ndarray) -> np.ndarray:
    if not isinstance(mat, np.ndarray):
        return compute_depol_elem(mat, true_mat)
    depols = np.zeros_like(mat)
    for ii in range(mat.shape[0]):
        for jj in range(mat.shape[1]):
            noisy = mat[ii, jj]
            ideal = true_mat[ii, jj]
            depol = compute_depol_elem(noisy, ideal)
            depols[ii, jj] = depol
    return depols


def rebuild_diag(m1: np.ndarray, d: np.ndarray):
    return m1 + np.diag(d)


def plot_depol_megamesh(edm: ExperimentDataManager, fpath: Path):
    l_edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    jobj = l_edm.load_last_saved_data_file

    textcolors = ["k", "w"]
    provider = jobj["inputs"]["provider_name"]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 1), constrained_layout=True)
    nq = jobj["inputs"]["n_qubits"]
    n_shots = jobj["inputs"]["n_shots"]
    n_qubits = jobj["inputs"]["n_qubits"]
    n_reps = jobj["inputs"]["n_reps"]

    which_obs = {"": r"\tilde{H}", "parity_": "Z" * n_qubits}

    offdiag_dict = {
        "+": r"\gamma_1",
        "-": r"\gamma_2",
        "-i": r"\gamma_3",
        "+i": r"\gamma_4",
    }

    for ind_x, ob in enumerate(which_obs.keys()):
        for ind_y, k in enumerate(offdiag_dict.keys()):
            mat = jobj["results"][f"offd_{ob}vals"][k]
            _diag = jobj["results"][f"diag_{ob}vals"]

            true_mat = jobj["results"][f"offd_{ob}true_vals"][k]
            true_diag = jobj["results"][f"diag_{ob}true_vals"]

            mat = rebuild_diag(mat, _diag)
            true_mat = rebuild_diag(true_mat, true_diag)

            cc = compute_depol(mat, true_mat)
            # check if depol can be measured at all
            cc[np.where(np.abs(mat) < 1 / np.sqrt(n_shots))] = np.nan

            color_cells = cc

            ax: plt.Axes = axes[ind_x, ind_y]
            im = ax.imshow(color_cells)
            threshold = im.norm(
                (np.max(color_cells) - np.min(color_cells)) / 2 + np.min(color_cells)
            )

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    sign = "" if mat[i, j].imag >= 0 else "-"

                    u1 = f"{mat[i, j]:.3f}"
                    u2 = f"{true_mat[i, j]:.3f}"

                    s = f"{u1}\n{sign}{u2}"

                    text = ax.text(
                        j,
                        i,
                        s,
                        ha="center",
                        va="center",
                        color=textcolors[int(im.norm(color_cells[i, j]) < threshold)],
                        fontsize=7,
                    )
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.grid(False)
            ax.set_title(rf"${offdiag_dict[k]}, {which_obs[ob]}$")

        ax.set_xticks(range(nq), range(1, 1 + nq))
        ax.set_yticks(range(nq), range(1, 1 + nq))

    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0.01, wspace=0.5)
    edm.var_dump(**jobj["inputs"], fpath=fpath)
    edm.save_figure(
        fig,
        f"mesh_depol_{provider}_{n_reps}",
        fig_shape="page-wide",
        add_timestamp=False,
    )

    plt.show()


if __name__ == "__main__":
    fpath = "hmatrix"
    edm = ExperimentDataManager("depol_megamesh", dry_run=False, project="nlce")
    plot_depol_megamesh(edm, fpath)
