from data_manager import ExtendedJSONDecoder

import io
import re
import json

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.qcomp_nlce.utils import DATA_DIR, reorder_matrix_elems
from src.qcomp_nlce.utils.results import pretty_print, pretty_print_diff
from src.qcomp_nlce.utils.workflow_utils import apply_symmetries
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


def within_error_cpx(c1: complex, c2: complex, std: complex) -> bool:
    if np.abs(c1.real - c2.real) <= std.real and np.abs(c1.imag - c2.imag) <= std.imag:
        return True
    else:
        return False


def within_error(c1: float, c2: float, std: float) -> bool:
    if np.abs(c1 - c2) <= std:
        return True
    else:
        return False


def plot_omatrix_depol_megamesh(edm: ExperimentDataManager):
    fpath = "omatrix"
    l_edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    jobj = l_edm.load_last_saved_data_file

    textcolors = ["k", "w"]
    provider = jobj["inputs"]["provider_name"]

    fig, axes = plt.subplots(ncols=2, constrained_layout=True)

    nq = jobj["inputs"]["n_qubits"]
    n_shots = jobj["inputs"]["n_shots"]
    n_qubits = jobj["inputs"]["n_qubits"]
    n_reps = jobj["inputs"]["n_reps"]

    mat = jobj["results"]["vals"]

    # I'm so tired schi
    true_mat = jobj["results"]["true_vals"] * 1j

    mat_std = jobj["results"]["stds"]

    mat, _ = apply_symmetries(reorder_matrix_elems(mat), mat, "vqe")
    mat_std = reorder_matrix_elems(mat_std)

    print("vals")
    pretty_print(mat)
    print("true")
    pretty_print(true_mat)
    print("diff")
    pretty_print_diff(mat, true_mat)
    print("std")
    pretty_print(mat_std)

    for jj, fn in enumerate((np.real, np.imag)):
        color_cells = np.abs(fn(true_mat) - fn(mat)) / fn(mat_std)

        print(f"error, {fn.__name__}")
        pretty_print(color_cells)

        ax: plt.Axes = axes[jj]
        im = ax.imshow(color_cells)
        threshold = im.norm(
            (np.max(color_cells) - np.min(color_cells)) / 2 + np.min(color_cells)
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                s = (
                    "Yes"
                    if within_error(mat[i, j], true_mat[i, j], mat_std[i, j])
                    else "No"
                )

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
        ax.set_title(rf"$O_{{ij}}$, {fn.__name__}")

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


def plot_hmatrix_depol_megamesh(edm: ExperimentDataManager):
    fpath = "hmatrix"
    l_edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    jobj = l_edm.load_last_saved_data_file

    textcolors = ["k", "w"]
    provider = jobj["inputs"]["provider_name"]

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 1), constrained_layout=True)
    nq = jobj["inputs"]["n_qubits"]
    n_shots = jobj["inputs"]["n_shots"]
    n_qubits = jobj["inputs"]["n_qubits"]
    n_reps = jobj["inputs"]["n_reps"]

    which_obs = {"": r"\tilde{H}"}

    offdiag_dict = {
        "+": r"\gamma_1",
        "-": r"\gamma_2",
        "-i": r"\gamma_3",
        "+i": r"\gamma_4",
    }

    for ind_y, k in enumerate(offdiag_dict.keys()):
        mat = jobj["results"]["offd_vals"][k]
        _diag = jobj["results"]["diag_vals"]

        true_mat = jobj["results"]["offd_true_vals"][k]
        true_diag = jobj["results"]["diag_true_vals"]

        mat_std = jobj["results"]["offd_stds"][k]
        _diag_std = jobj["results"]["diag_stds"]

        mat = rebuild_diag(mat, _diag)
        true_mat = rebuild_diag(true_mat, true_diag)
        mat_std = rebuild_diag(mat_std, _diag_std)
        color_cells = np.abs(true_mat - mat) / np.abs(mat_std)

        print(f"{k}, error")
        pretty_print(color_cells)
        print(f"{k}, std")
        pretty_print(mat_std)

        ax: plt.Axes = axes[ind_y]
        im = ax.imshow(color_cells)
        threshold = im.norm(
            (np.max(color_cells) - np.min(color_cells)) / 2 + np.min(color_cells)
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                s = (
                    "Yes"
                    if within_error(mat[i, j], true_mat[i, j], mat_std[i, j])
                    else "No"
                )

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
        ax.set_title(rf"${offdiag_dict[k]}, \tilde{{H}}$")

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
    edm = ExperimentDataManager(
        "omatrix_within_std_megamesh", dry_run=True, project="nlce"
    )
    plot_omatrix_depol_megamesh(edm)
