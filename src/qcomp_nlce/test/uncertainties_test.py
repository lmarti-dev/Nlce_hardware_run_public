from uncertainties import unumpy, ufloat, Variable
import numpy as np

import matplotlib.pyplot as plt


from qcomp_nlce.utils import (
    u_H_effective,
    H_effective,
    dispersion_1d_thermodynamic,
    dispersion_1d_thermodynamic_ks_UT,
    unscented_transform,
)


def test_func(arr, k):

    y = np.linspace(0.1, 0.2, len(arr))

    return np.dot(arr, y) * np.cos(k)


def monte_carlo_like(n_samples: int, n_qubits_max: int):
    ks = np.linspace(0, np.pi, 20)
    out_vals = np.zeros(ks, n_samples)
    for m in n_samples:
        expec_l, _, overl_l, _, gs_l, _ = random_heff_err(n_qubits_max)
        expec_s, _, overl_s, _, gs_s, _ = random_heff_err(n_qubits_max - 1)
        H_eff_Nb = H_effective(expec_l, gs_l, overl_l)
        H_eff_Ns = H_effective(expec_s, gs_s, overl_s)
        out_vals[:, m] = [
            dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks
        ]


def random_heff_err(n_qubits):
    expec = np.random.rand(n_qubits, n_qubits) + 1j * np.random.rand(n_qubits, n_qubits)
    overl = np.random.rand(n_qubits, n_qubits) + 1j * np.random.rand(n_qubits, n_qubits)
    gs = np.random.rand(1)

    fac = 1e15

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
        expectations_std_s=expec_s,
        gs_expectation_s=gs_s,
        gs_expectation_std_s=gs_std_s,
    )

    fig, ax = plt.subplots()
    ax.errorbar(ks, means.flatten(), yerr=Pmats.flatten(), label="filter")
    ax.plot(ks, vals, label="true")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    ...
