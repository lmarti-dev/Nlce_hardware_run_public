from uncertainties import unumpy, ufloat, Variable
import numpy as np

import matplotlib.pyplot as plt


from qcomp_nlce.utils import u_H_effective, H_effective, dispersion_1d_thermodynamic


def arr_to_vec(m: np.ndarray) -> np.ndarray:
    return np.reshape(m, np.size(m))


def vec_to_arr(m: np.ndarray) -> np.ndarray:
    return np.reshape(m, (int(np.sqrt(len(m))), int(np.sqrt(len(m)))))


def to_mega_arrays(
    overlap_l: np.ndarray,
    overlap_std_l: np.ndarray,
    expectations_l: np.ndarray,
    expectations_std_l: np.ndarray,
    gs_expectation_l: float,
    gs_expectation_std_l: float,
    overlap_s: np.ndarray,
    overlap_std_s: np.ndarray,
    expectations_s: np.ndarray,
    expectations_std_s: np.ndarray,
    gs_expectation_s: float,
    gs_expectation_std_s: float,
):

    arr = np.concatenate(
        list(
            arr_to_vec(m)
            for m in (
                overlap_l,
                expectations_l,
                np.array(gs_expectation_l),
                overlap_s,
                expectations_s,
                np.array(gs_expectation_s),
            )
        )
    )

    stds = np.concatenate(
        list(
            arr_to_vec(m)
            for m in (
                overlap_std_l,
                expectations_std_l,
                np.array(gs_expectation_std_l),
                overlap_std_s,
                expectations_std_s,
                np.array(gs_expectation_std_s),
            )
        )
    )

    return arr, stds


def unroll_arr(arr, nqm: int):

    sml = int(nqm**2)
    sms = int((nqm - 1) ** 2)

    overlap_l = vec_to_arr(arr[0:sml])
    expectations_l = vec_to_arr(arr[sml : 2 * sml])
    gs_expectation_l = arr[2 * sml : 2 * sml + 1][0]

    step = 2 * sml + 1

    overlap_s = vec_to_arr(arr[step : step + sms])
    expectations_s = vec_to_arr(arr[step + sms : step + 2 * sms])
    gs_expectation_s = arr[step + 2 * sms : step + 2 * sms + 1][0]

    return (
        overlap_l,
        expectations_l,
        gs_expectation_l,
        overlap_s,
        expectations_s,
        gs_expectation_s,
    )


def dispersion_from_mats(arr, k, nqm: int) -> float:
    (
        overlap_l,
        expectations_l,
        gs_expectation_l,
        overlap_s,
        expectations_s,
        gs_expectation_s,
    ) = unroll_arr(arr, nqm)

    H_eff_Nb = H_effective(gs_expectation_l, expectations_l, overlap_l)
    H_eff_Ns = H_effective(gs_expectation_s, expectations_s, overlap_s)

    return dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns)


def unscented_transform(
    func: callable,
    vals: np.ndarray,
    stds: np.ndarray,
    alpha: float = 1e-2,
    kappa: float = 0,
    beta: float = 2,
):
    # brought to you by
    # The Unscented Kalman Filter for Nonlinear Estimation
    # Eric A. Wan and Rudolph van der Merwe
    # and my desire to end it all

    # anyway there is no covariance between the matrix variables. Hopefully

    test = func(vals)
    ll = test.shape
    if ll == ():
        ll = 1
    L = len(vals)
    lam = (alpha**2) * (L + kappa)

    Xs = np.zeros((len(vals), 2 * L + 1), dtype=complex)
    Ys = np.zeros((ll, 2 * L + 1), dtype=complex)
    Wms = np.zeros((2 * L + 1), dtype=complex)
    Wcs = np.zeros((2 * L + 1), dtype=complex)

    Wms[0] = lam / (L + lam)
    Wcs[0] = (beta + 1 - alpha**2) + lam / (L + lam)

    Xs[:, 0] = vals
    Ys[:, 0] = func(Xs[:, 0])

    stdsscaled = np.sqrt(L + lam) * stds

    for m in range(1, 2 * L + 1):
        Wms[m] = 1 / (2 * (L + lam))
        Wcs[m] = 1 / (2 * (L + lam))

    for m in range(L):
        Xs[:, m + 1] = vals
        Xs[m, m + 1] += stdsscaled[m]

        Ys[:, m + 1] = func(Xs[:, m + 1])

    for m in range(L, 2 * L):
        Xs[:, m + 1] = vals
        Xs[m - L, m + 1] -= stdsscaled[m - L]
        Ys[:, m + 1] = func(Xs[:, m + 1])

    ymean = Ys @ Wms

    Pmat = np.zeros((ll, ll), dtype=complex)
    for m in range(2 * L + 1):
        Pmat += Wcs[m] * (np.outer(Ys[:, m] - ymean, Ys[:, m] - ymean))
    return ymean, Pmat


def test_func(arr, k):

    y = np.linspace(0.1, 0.2, len(arr))

    return np.dot(arr, y) * np.cos(k)


def uf_dispersion_1d_thermodynamic_ks(
    ks: np.ndarray,
    overlap_l: np.ndarray,
    overlap_std_l: np.ndarray,
    expectations_l: np.ndarray,
    expectations_std_l: np.ndarray,
    gs_expectation_l: float,
    gs_expectation_std_l: float,
    overlap_s: np.ndarray,
    overlap_std_s: np.ndarray,
    expectations_s: np.ndarray,
    expectations_std_s: np.ndarray,
    gs_expectation_s: float,
    gs_expectation_std_s: float,
):
    arr, stds = to_mega_arrays(
        overlap_l,
        overlap_std_l,
        expectations_l,
        expectations_std_l,
        gs_expectation_l,
        gs_expectation_std_l,
        overlap_s,
        overlap_std_s,
        expectations_s,
        expectations_std_s,
        gs_expectation_s,
        gs_expectation_std_s,
    )
    means = []
    Pmats = []

    nqm = overlap_l.shape[0]

    vals = []

    for k in ks:

        def func(x):
            # return test_func(x, k)
            return dispersion_from_mats(x, k, nqm)

        ymean, Pmat = unscented_transform(
            func,
            arr,
            stds,
        )
        means.append(ymean)
        Pmats.append(Pmat)
        H_eff_Nb = H_effective(gs_expectation_l, expectations_l, overlap_l)
        H_eff_Ns = H_effective(gs_expectation_s, expectations_s, overlap_s)
        vals.append(dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns))
    return np.array(vals), np.array(means), np.array(Pmats)


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
    expec = np.random.rand(
        n_qubits, n_qubits
    )  # + 1j * np.random.rand(n_qubits, n_qubits)
    overl = np.random.rand(
        n_qubits, n_qubits
    )  # + 1j * np.random.rand(n_qubits, n_qubits)
    gs = np.random.rand(1)

    expec_std = np.random.normal(size=(n_qubits, n_qubits)) / 1000
    overl_std = np.random.normal(size=(n_qubits, n_qubits)) / 1000
    gs_std = np.random.normal() / 1000

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

    vals, means, Pmats = uf_dispersion_1d_thermodynamic_ks(
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
    filter_workflow()
