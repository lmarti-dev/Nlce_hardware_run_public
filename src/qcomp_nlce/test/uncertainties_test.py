import numpy as np

import matplotlib.pyplot as plt
import json

from qcomp_nlce.utils import (
    H_effective,
    dispersion_1d_thermodynamic,
    dispersion_1d_thermodynamic_ks_UT,
    exact_dispersion_1qp_1d,
)
from pathlib import Path
from data_manager import ExtendedJSONDecoder


def test_func(arr, k):

    y = np.linspace(0.1, 0.2, len(arr))

    return np.dot(arr, y) * np.cos(k)


def monte_carlo_like(n_samples: int, n_qubits_max: int):
    ks = np.linspace(0, np.pi, 20)
    out_vals = np.zeros(ks, n_samples)
    for m in range(n_samples):
        expec_l, _, overl_l, _, gs_l, _ = random_heff_err(n_qubits_max)
        expec_s, _, overl_s, _, gs_s, _ = random_heff_err(n_qubits_max - 1)
        H_eff_Nb = H_effective(expec_l, gs_l, overl_l)
        H_eff_Ns = H_effective(expec_s, gs_s, overl_s)
        out_vals[:, m] = [
            dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks
        ]

    fig, ax = plt.subplots()
    for m in range(n_samples):
        ax.plot(ks, out_vals[:, m])
    exact_dispersion_1qp_1d(1, ks)
    plt.show()


def cn2cn(loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.random.normal(loc=loc.real, scale=scale.real)
    y = np.random.normal(loc=loc.imag, scale=scale.imag)
    return x + 1j * y


def load_res_json(fpath: Path = None):

    if fpath is None:
        fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_10\workflow_tfim_hadam_stefan_AQT_2000shots_00004\run_00000\data\green_00000.json"
        n_shots = 2000
        fac = np.sqrt(n_shots)
    with open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)

    be = "fake_backend"

    data_l = jobj["results"]["vqe"]["dict_l"]
    data_s = jobj["results"]["vqe"]["dict_s"]

    expect_l = data_l["exd"][be]["values"]
    expect_std_l = data_l["exd"][be]["stds"] / fac
    overlap_l = data_l["eld"][be]["values"]
    overlap_std_l = data_l["eld"][be]["stds"] / fac
    gs_l = data_l["gsd"][be]["energy"]
    gs_std_l = data_l["gsd"][be]["std"] / fac

    expect_s = data_s["exd"][be]["values"]
    expect_std_s = data_s["exd"][be]["stds"] / fac
    overlap_s = data_s["eld"][be]["values"]
    overlap_std_s = data_s["eld"][be]["stds"] / fac
    gs_s = data_s["gsd"][be]["energy"]
    gs_std_s = data_s["gsd"][be]["std"] / fac

    return (
        expect_l,
        expect_std_l,
        overlap_l,
        overlap_std_l,
        gs_l,
        gs_std_l,
        expect_s,
        expect_std_s,
        overlap_s,
        overlap_std_s,
        gs_s,
        gs_std_s,
    )


def get_UT_disp_vals(
    ks: np.ndarray,
    fpath: Path = None,
):

    (
        expect_l,
        expect_std_l,
        overlap_l,
        overlap_std_l,
        gs_l,
        gs_std_l,
        expect_s,
        expect_std_s,
        overlap_s,
        overlap_std_s,
        gs_s,
        gs_std_s,
    ) = load_res_json()

    vals, means, Pmats = dispersion_1d_thermodynamic_ks_UT(
        ks,
        overlap_l=overlap_l,
        overlap_std_l=overlap_std_l,
        expectations_l=expect_l,
        expectations_std_l=expect_std_l,
        gs_expectation_l=gs_l,
        gs_expectation_std_l=gs_std_l,
        overlap_s=overlap_s,
        overlap_std_s=overlap_std_s,
        expectations_s=expect_s,
        expectations_std_s=expect_std_s,
        gs_expectation_s=gs_s,
        gs_expectation_std_s=gs_std_s,
    )

    return means, Pmats


def get_heff_mc_sampler() -> callable:

    (
        expect_l,
        expect_std_l,
        overlap_l,
        overlap_std_l,
        gs_l,
        gs_std_l,
        expect_s,
        expect_std_s,
        overlap_s,
        overlap_std_s,
        gs_s,
        gs_std_s,
    ) = load_res_json()

    def sampler(ks, fac: float = 1.0):
        sa_expect_l = cn2cn(loc=expect_l, scale=expect_std_l * fac)
        sa_overlap_l = cn2cn(loc=overlap_l, scale=overlap_std_l * fac)
        sa_gs_l = cn2cn(loc=gs_l, scale=gs_std_l * fac)

        sa_expect_s = cn2cn(loc=expect_s, scale=expect_std_s * fac)
        sa_overlap_s = cn2cn(loc=overlap_s, scale=overlap_std_s * fac)
        sa_gs_s = cn2cn(loc=gs_s, scale=gs_std_s * fac)

        H_eff_Nb = H_effective(sa_gs_l, sa_expect_l, sa_overlap_l)
        H_eff_Ns = H_effective(sa_gs_s, sa_expect_s, sa_overlap_s)

        return [dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks]

    return sampler


def random_heff_err(n_qubits):
    expec = np.random.rand(n_qubits, n_qubits) + 1j * np.random.rand(n_qubits, n_qubits)
    overl = np.random.rand(n_qubits, n_qubits) + 1j * np.random.rand(n_qubits, n_qubits)
    gs = np.random.rand(1)

    fac = 1e1

    expec_std = np.random.normal(size=(n_qubits, n_qubits)) / fac
    overl_std = np.random.normal(size=(n_qubits, n_qubits)) / fac
    gs_std = np.random.normal() / fac

    return expec, expec_std, overl, overl_std, gs, gs_std


def filter_workflow():

    n_qubits_max = 5

    expec_l, expec_std_l, overl_l, overl_std_l, gs_l, gs_std_l = random_heff_err(
        n_qubits_max
    )
    expec_s, expec_std_s, overl_s, overl_std_s, gs_s, gs_std_s = random_heff_err(
        n_qubits_max - 1
    )
    ks = np.linspace(0, np.pi, 100)

    vals, means, Pmats = dispersion_1d_thermodynamic_ks_UT(
        ks,
        overlap_l=overl_l,
        overlap_std_l=overl_std_l,
        expectations_l=expec_l,
        expectations_std_l=expec_std_l,
        gs_expectation_l=gs_l,
        gs_expectation_std_l=gs_std_l,
        overlap_s=overl_s,
        overlap_std_s=overl_std_s,
        expectations_s=expec_s,
        expectations_std_s=expec_std_s,
        gs_expectation_s=gs_s,
        gs_expectation_std_s=gs_std_s,
    )

    fig, ax = plt.subplots()
    ax.errorbar(ks, means.flatten(), yerr=Pmats.flatten(), label="filter")
    ax.plot(ks, vals, label="true")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    sampler = get_heff_mc_sampler()
    n_samples = 10000
    ks = np.linspace(0, np.pi, 100)

    fig, ax = plt.subplots()
    # for n in range(n_samples):
    #     ax.plot(ks, sampled[:, n], alpha=0.01)

    ax.plot(ks, sampler(ks, 0.0), "kd:", label="Measured")
    ax.plot(ks, exact_dispersion_1qp_1d(1, ks), "rx--", label="Analytic")
    # means, pmats = get_UT_disp_vals(ks)
    # ax.errorbar(ks, means, pmats, color="g", label="UT")

    sampled = np.zeros((len(ks), n_samples))
    for n in range(n_samples):
        print(f"{n}/{n_samples}", end="\r")
        sampled[:, n] = sampler(ks)
    ax.errorbar(
        ks,
        np.mean(sampled, axis=1),
        np.std(sampled, axis=1),
        color="blue",
        label="Monte Carlo",
    )
    ax.legend()
    plt.show()
