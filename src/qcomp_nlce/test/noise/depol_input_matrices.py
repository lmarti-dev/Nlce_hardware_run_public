from src.qcomp_nlce.utils import (
    dispersion_1d_thermodynamic,
    DATA_DIR,
    cluster_dispersion,
    H_effective,
    exact_dispersion_1qp_1d,
)

from data_manager import ExperimentDataManager

import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt


def load_vqe_mats() -> dict:
    ledm = ExperimentDataManager.load(Path(DATA_DIR, "1qp/vqe_tfim_80j_0hl_5q"))

    jobj = ledm.load_last_saved_data_file

    d = {
        "dict_l": {"eld": None, "exd": None, "gsd": None},
        "dict_s": {"eld": None, "exd": None, "gsd": None},
    }

    for k in ("dict_l", "dict_s"):
        d[k]["eld"] = jobj["results"]["vqe"][k]["eld"]["statevector"]["values"]
        d[k]["exd"] = jobj["results"]["vqe"][k]["exd"]["statevector"]["values"]
        d[k]["gsd"] = jobj["results"]["vqe"][k]["gsd"]["statevector"]["energy"]

    return d


def dispersion_from_d(ks: np.ndarray, d: dict) -> np.ndarray:
    heff_l = H_effective(d["dict_l"]["gsd"], d["dict_l"]["exd"], d["dict_l"]["eld"])
    heff_s = H_effective(d["dict_s"]["gsd"], d["dict_s"]["exd"], d["dict_s"]["eld"])
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


def dispersion_from_d_dilate(
    ks: np.ndarray, d: dict, s1: float, s2: float
) -> np.ndarray:
    heff_l = H_effective(
        s1 * d["dict_l"]["gsd"], d["dict_l"]["exd"], d["dict_l"]["eld"]
    )
    heff_s = H_effective(
        s2 * d["dict_s"]["gsd"], d["dict_s"]["exd"], d["dict_s"]["eld"]
    )
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


def dispersion_from_squish_move(
    ks: np.ndarray, d: dict, pe: float, ph: float
) -> np.ndarray:
    heff_l = H_effective(
        (1 - pe) * d["dict_l"]["gsd"], (1 - ph) * d["dict_l"]["exd"], d["dict_l"]["eld"]
    )
    heff_s = H_effective(d["dict_s"]["gsd"], d["dict_s"]["exd"], d["dict_s"]["eld"])
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


def cluster_from_d_add(ks: np.ndarray, d: dict, s: float):
    N = len(d["dict_l"]["exd"])
    heff_l = (
        H_effective(d["dict_l"]["gsd"], d["dict_l"]["exd"], d["dict_l"]["eld"])
        - s * np.eye(N) * d["dict_l"]["gsd"]
    )
    return [cluster_dispersion(N, heff_l, k) for k in ks]


def depol_mats(
    p1: float,
    p2: float,
    which: list[str] = ["exd", "eld", "gsd"],
):
    d = load_vqe_mats()
    for k in ("dict_l",):
        for m in which:
            if m == "gsd":
                d[k][m] = (1 - p1) * d[k][m]
            else:
                d[k][m] = (1 - p2) * d[k][m]

        # print(d[k]["gsd"] / np.mean(d[k]["exd"]))

    return d


def plot_depol_ratio():
    fig, ax = plt.subplots()
    ks = np.linspace(0, np.pi, 21)
    n_qubits_max = 5
    J = 0.8
    hl = 0
    p_h = 0.1

    dddp = depol_mats(0, 0)
    disp = dispersion_from_d(ks, dddp)

    ax.plot(ks, disp, "x-", label="no noise")

    for npow in np.linspace(-2, 0, 10):
        p_e = 10**npow * p_h
        dddp = depol_mats(p_e, p_h)
        disp = dispersion_from_d(ks, dddp)

        ax.plot(ks, disp, label=rf"$p_E/p_H$: {p_e / p_h:.3f}")
    ax.legend()
    plt.show()


def plot_delta_in_e():
    fig, ax = plt.subplots()
    ks = np.linspace(0, np.pi, 21)
    dddp = depol_mats(0, 0)
    disp = dispersion_from_d(ks, dddp)

    ax.plot(ks, disp, "x-", label="no noise")

    for npow1 in np.linspace(0, 0.1, 3):
        s1 = 10**npow1
        for npow2 in np.linspace(0, 0.1, 3):
            s2 = 10**npow2
            dddp = depol_mats(0, 0)
            disp = dispersion_from_d_dilate(ks, dddp, s1, s2)

            ax.plot(ks, disp, label=rf"{s1:.3f}, {s2:.3f}")
    ax.legend()
    plt.show()


def frm10(num):
    exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), 2)
    return rf"{coeff:.2f}\cdot10^{{{exponent:n}}}"


def plot_squish_move():
    fig, ax = plt.subplots()
    ks = np.linspace(0, np.pi, 21)
    dddp = depol_mats(0, 0)
    disp = dispersion_from_d(ks, dddp)

    ax.plot(ks, disp, "x-", label="no noise")

    eddisp = [exact_dispersion_1qp_1d(0.8, k) for k in ks]
    ax.plot(ks, eddisp, "dr:", label="Analytic")

    ax.plot(ks, disp, "x-", label="no noise")

    pe = 1e-10
    ph = 0.06
    dddp = depol_mats(0, 0)
    disp = dispersion_from_squish_move(ks, dddp, pe, ph)

    ax.plot(ks, disp, label=rf"$p_E: {frm10(pe)}, p_H: {frm10(ph)}$")
    ax.legend()
    plt.show()


def plot_cluster_dilate():
    fig, ax = plt.subplots()
    ks = np.linspace(0, np.pi, 21)
    dddp = depol_mats(0, 0)
    disp = cluster_from_d_add(ks, dddp, 0)

    ax.plot(ks, disp, "x-", label="no noise")

    for npow in np.linspace(-2, -1, 3):
        s = 10**npow
        dddp = depol_mats(0, 0)
        disp = cluster_from_d_add(ks, dddp, s)

        ax.plot(ks, disp, label=rf"{s:.3f}")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot_squish_move()
