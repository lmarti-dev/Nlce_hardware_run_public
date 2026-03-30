from data_manager import ExtendedJSONDecoder

import io
import re
import json

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal
from src.qcomp_nlce.utils import BACKEND_TYPES, DATA_DIR, ALGORITHMS_ALIAS
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
    which_algorithm: Literal["vqe", "sweep"],
):
    return (
        jobj["results"][which_algorithm][size][which_matrix][which_backend]["values"],
        jobj["results"][which_algorithm][size][which_matrix][which_backend]["stds"],
    )


def get_true_values(
    jobj: dict,
    size: Literal["dict_l", "dict_s"],
    which_matrix: Literal["exs", "eld"],
    which_algorithm: Literal["vqe", "sweep"],
):
    return jobj["results"][which_algorithm][size][which_matrix]["system"]["values"]


def get_gs(
    jobj: dict,
    size: Literal["dict_l", "dict_s"],
    which_backend: BACKEND_TYPES,
    which_algorithm: Literal["vqe", "sweep"],
):
    return (
        jobj["results"][which_algorithm][size]["gsd"][which_backend]["energy"],
        jobj["results"][which_algorithm][size]["gsd"][which_backend]["std"],
    )


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


def complex_depol(mat: np.ndarray, true_mat: np.ndarray) -> np.ndarray:
    return compute_depol(mat.real, true_mat.real) + 1j * compute_depol(
        mat.imag, true_mat.imag
    )


SHOWABLE = Literal["depol", "error", "std", "value", "errorpercent"]


def plot_matrix_breakdown(
    edm: ExperimentDataManager,
    item: list[str, str, str],
    which_color: SHOWABLE,
    which_cell: SHOWABLE,
    show_std: bool = False,
):
    fpath = item[0]
    which_algorithm = item[1]
    which_backend = item[2]

    l_edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    jobj = l_edm.load_last_saved_data_file
    if "model_dir" in jobj["inputs"]:
        which_model = jobj["inputs"]["model_dir"]
    else:
        which_model = jobj["inputs"]["which_model"]

    textcolors = ["k", "w"]
    provider = jobj["inputs"]["provider_name"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), constrained_layout=True)
    pn = jobj["inputs"]["provider_name"]
    nq = jobj["inputs"]["n_qubits"]
    n_shots = jobj["inputs"]["n_shots"]
    n_reps = jobj["inputs"]["n_reps"]
    n_shots *= n_reps

    SIZES_D = {"dict_l": f"{nq} qubits", "dict_s": f"{nq - 1} qubits"}
    WHICH_D = {
        "eld": r"$\langle j | U | k \rangle$",
        "exd": r"$\langle j | U^\dagger H U | k \rangle$",
    }

    for ind_x, size in enumerate(("dict_l", "dict_s")):
        for ind_y, which_mat in enumerate(("eld", "exd")):
            mat, stds = get_matrix(
                jobj, size, which_mat, which_backend, which_algorithm
            )
            mat = apply_symmetry(mat, which_mat, which_algorithm)
            true_mat = get_true_values(jobj, size, which_mat, which_algorithm)
            stds /= np.sqrt(n_shots)
            nq = len(mat)
            if which_color == "depol":
                cc = compute_depol(np.abs(mat), np.abs(true_mat))
                # check if depol can be measured at all
                cc[np.where(np.abs(mat) < 1 / np.sqrt(n_shots))] = np.nan

                color_cells = cc
            elif which_color == "error":
                color_cells = np.abs(true_mat - mat)
            elif which_color == "errorpercent":
                color_cells = np.abs(true_mat - mat) / np.abs(mat)
            elif which_color == "value":
                color_cells = np.abs(mat)

            elif which_color == "std":
                color_cells = np.abs(stds)
            else:
                raise ValueError(f"Expected valid which_color, got: {which_color}")
            ax: plt.Axes = axes[ind_x, ind_y]
            im = ax.imshow(color_cells)
            threshold = im.norm((np.max(color_cells) - np.min(color_cells))) / 2

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if which_cell == "value":
                        sign = "" if mat[i, j].imag >= 0 else "-"
                        if show_std:
                            u1 = parnum(mat[i, j].real, stds[i, j].real)
                            u2 = parnum(np.abs(mat[i, j].imag), stds[i, j].imag)
                        else:
                            u1 = f"{mat[i, j].real:.3f}"
                            u2 = f"{mat[i, j].imag:.3f}"

                        s = f"{u1}\n{sign}{u2}"
                    elif which_cell == "std":
                        rstd = np.abs(stds[i, j].real)
                        istd = np.abs(stds[i, j].imag)
                        s = f"{float(rstd):.2f}\n{float(istd):.2f}"

                    elif which_cell == "depol":
                        rstd = np.abs(
                            compute_depol(mat.real[i, j], true_mat.real[i, j])
                        )
                        istd = np.abs(
                            compute_depol(mat.imag[i, j], true_mat.imag[i, j])
                        )
                        s = f"{float(rstd):.2f}\n{float(istd):.2f}"

                    elif which_cell == "error":
                        rstd = np.abs(mat.real - true_mat.real)
                        istd = np.abs(mat.imag - true_mat.imag)
                        s = f"{float(rstd):.2f}\n{float(istd):.2f}"

                    elif which_cell == "errorpercent":
                        rstd = np.abs(mat.real - true_mat.real) / np.abs(mat.real)
                        istd = np.abs(mat.imag - true_mat.imag) / np.abs(mat.imag)
                        s = f"{float(rstd):.2f}\n{float(istd):.2f}"

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
            ax.set_title(rf"{SIZES_D[size]} - {WHICH_D[which_mat]}")
            ax.grid(False)

        ax.set_xticks(range(nq), range(1, 1 + nq))
        ax.set_yticks(range(nq), range(1, 1 + nq))
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0.01, wspace=0.5)
    edm.var_dump(**jobj["inputs"], fpath=fpath)
    edm.save_figure(
        fig,
        f"{which_algorithm}_{which_backend}_{which_model}_{provider}_meshes_{which_color}c_{which_cell}t",
        fig_shape="double-size",
        add_timestamp=False,
    )


def apply_symmetry(
    mat: np.ndarray, which: Literal["exd", "eld"], alg: ALGORITHMS_ALIAS
):
    if which == "eld" and alg == "vqe":
        mat = (mat + mat[::-1, ::-1]) / 2

    elif which == "eld" and alg == "sweep":
        mat = (mat + mat[::-1, ::-1].T + mat[::-1, ::-1] + mat.T) / 4

    else:
        mat = (mat + mat[::-1, ::-1].T.conj() + mat.T.conj() + mat[::-1, ::-1]) / 4
    return mat


def parnum(val: float, std: float) -> str:
    if std == 0:
        return f"{val:.3f}"
    nz = re.match("0+", str(std).replace(".", "")).span()[-1] - 1
    s2 = str(std)[nz + 2 : nz + 4]
    s1 = f"{val:.{nz}f}({s2})"
    return s1


def real_runs():
    style()
    dry_run = False
    which_backend = "real_backend"

    TFIM_80j_sweep_fpath = r"1qp/sweep_tfim_80j_0hl_5q"
    TFIM_80j_vqe_fpath = r"1qp/vqe_tfim_80j_0hl_5q"
    TFIM_100j_vqe_fpath = r"1qp/vqe_tfim_100j_0hl_5q"
    TFIM_50j_10hl_vqe_fpath = r"1qp/vqe_tfim_50j_10hl_5q"
    TFIM_20j_ladder_vqe_fpath = r"1qp/vqe_tfim_20j_0hl_8q_ladder"

    edm = ExperimentDataManager(
        f"plot_meshes_{which_backend}",
        project="nlce",
        dry_run=dry_run,
    )

    for item in (
        (TFIM_80j_sweep_fpath, "sweep", which_backend),
        (TFIM_80j_vqe_fpath, "vqe", which_backend),
        (TFIM_100j_vqe_fpath, "vqe", which_backend),
        (TFIM_50j_10hl_vqe_fpath, "vqe", which_backend),
        (TFIM_20j_ladder_vqe_fpath, "vqe", which_backend),
    ):
        # plot_matrix_breakdown_error(edm, item)
        # plot_matrix_breakdown(edm, item)
        plot_matrix_breakdown(edm, item, "value", "depol")
        plot_matrix_breakdown(edm, item, "error", "value")

    # plt.show()


def fake_runs():
    style()
    dry_run = False
    which_backend = "fake_backend"

    better10 = r"depol_vs_shots/vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_10_2000shots_x_20reps"
    normal = (
        r"depol_vs_shots/vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_2000shots_x_20reps"
    )
    better100 = r"depol_vs_shots/vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_BETTER_100_2000shots_x_20reps"
    worse2 = r"depol_vs_shots/vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_2_2000shots_x_20reps"

    edm = ExperimentDataManager(
        f"plot_meshes_{which_backend}",
        project="nlce",
        dry_run=dry_run,
    )

    for item in (
        (normal, "vqe", which_backend),
        (better10, "vqe", which_backend),
        (better100, "vqe", which_backend),
        (worse2, "vqe", which_backend),
    ):
        plot_matrix_breakdown(edm, item, "value", "depol")
        plot_matrix_breakdown(edm, item, "error", "value")

    # plt.show()


if __name__ == "__main__":
    real_runs()
