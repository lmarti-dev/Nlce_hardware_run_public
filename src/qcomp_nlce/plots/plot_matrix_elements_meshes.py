from data_manager import ExtendedJSONDecoder

import io
import re
import json

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal
from src.qcomp_nlce.utils import BACKEND_TYPES, DATA_DIR
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


def plot_matrix_breakdown(edm: ExperimentDataManager, item: list[str, str, str]):
    fpath = item[0]
    which_algorithm = item[1]
    which_backend = item[2]
    jobj = load_json(Path(DATA_DIR, fpath))
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

    SIZES_D = {"dict_l": f"{nq} qubits", "dict_s": f"{nq - 1} qubits"}
    WHICH_D = {
        "eld": rf"$\langle j | U | k \rangle$",
        "exd": rf"$\langle j | U^\dagger H U | k \rangle$",
    }

    for ind_x, size in enumerate(("dict_l", "dict_s")):
        for ind_y, which in enumerate(("eld", "exd")):
            mat, stds = get_matrix(jobj, size, which, which_backend, which_algorithm)
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
    edm.var_dump(**jobj["inputs"], fpath=fpath)
    edm.save_figure(
        fig,
        f"{which_algorithm}_{which_backend}_{which_model}_meshes_std",
        fig_shape="double-size",
        add_timestamp=False,
    )


def parnum(val: float, std: float) -> str:
    if std == 0:
        return f"{val:.3f}"
    nz = re.match("0+", str(std).replace(".", "")).span()[-1] - 1
    s2 = str(std)[nz + 2 : nz + 4]
    s1 = f"{val:.{nz}f}({s2})"
    return s1


def plot_matrix_breakdown_error(edm: ExperimentDataManager, item: list[str, str, str]):
    fpath = item[0]
    which_algorithm = item[1]
    which_backend = item[2]
    jobj = load_json(Path(DATA_DIR, fpath))
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

    SIZES_D = {"dict_l": f"{nq} qubits", "dict_s": f"{nq - 1} qubits"}
    WHICH_D = {
        "eld": rf"$\langle j | U | k \rangle$",
        "exd": rf"$\langle j | U^\dagger H U | k \rangle$",
    }

    for ind_x, size in enumerate(("dict_l", "dict_s")):
        for ind_y, which_mat in enumerate(("eld", "exd")):
            mat, stds = get_matrix(
                jobj, size, which_mat, which_backend, which_algorithm
            )
            true_mat = get_true_values(jobj, size, which_mat, which_algorithm)
            stds /= np.sqrt(n_shots)
            nq = len(mat)
            astds = np.abs(true_mat - mat)
            ax: plt.Axes = axes[ind_x, ind_y]
            im = ax.imshow(np.abs(astds))
            threshold = im.norm(np.max(astds)) / 2

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ttype = "val"
                    if ttype == "val":
                        sign = "" if mat[i, j].imag >= 0 else "-"
                        u1 = parnum(mat[i, j].real, stds[i, j].real)
                        u2 = parnum(np.abs(mat[i, j].imag), stds[i, j].imag)

                        s = f"{u1}\n{sign}{u2}"
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
        f"{which_algorithm}_{which_backend}_{which_model}_meshes_error",
        fig_shape="double-size",
        add_timestamp=False,
    )


if __name__ == "__main__":
    style()
    dry_run = False
    which_backend = "statevector"

    TFIM_80j_sweep_fpath = (
        r"1qp/sweep_tfim_80j_0hl_5q/run_00000/data/chestnuts_00000.json"
    )
    TFIM_80j_vqe_fpath = r"1qp/vqe_tfim_80j_0hl_5q/run_00000/data/cantaloupe_00000.json"
    TFIM_100j_vqe_fpath = r"1qp/vqe_tfim_100j_0hl_5q/run_00000/data/peanuts_00000.json"
    TFIM_50j_10hl_vqe_fpath = (
        r"1qp/vqe_tfim_50j_10hl_5q/run_00000/data/pears_00000.json"
    )

    TFIM_20j_ladder_vqe_fpath = (
        r"1qp/vqe_tfim_20j_0hl_8q_ladder/run_00000/data/plantains_00000.json"
    )

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
        plot_matrix_breakdown_error(edm, item)
        plot_matrix_breakdown(edm, item)

    # plt.show()
