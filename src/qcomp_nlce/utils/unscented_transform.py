from qcomp_nlce.utils.effective_hamiltonian import (
    H_effective,
)
from qcomp_nlce.utils.dispersion import dispersion_1d_thermodynamic
import numpy as np
from typing import Literal


def arr_to_vec(m: np.ndarray) -> np.ndarray:

    v1 = m.real
    v2 = m.imag

    return np.concatenate([np.reshape(v1, np.size(v1)), np.reshape(v2, np.size(v2))])


def vec_to_arr(m: np.ndarray) -> np.ndarray:

    n = int(len(m) / 2)

    assert len(m) % 2 == 0
    assert np.isclose(np.sqrt(n), np.round(np.sqrt(n)))

    v1 = m[:n]
    v2 = m[n:]

    a = np.reshape(v1, (int(np.sqrt(len(v1))), int(np.sqrt(len(v1)))))
    b = np.reshape(v2, (int(np.sqrt(len(v2))), int(np.sqrt(len(v2)))))

    return a + 1j * b


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


def complex_to_double_real(z: np.ndarray) -> np.ndarray:
    x = z.real
    y = z.imag

    if isinstance(x, float) and isinstance(y, float):
        return np.array([x, y])

    return np.concatenate([x, y])


def double_real_to_complex(r: np.ndarray) -> np.ndarray:
    assert len(r) % 2 == 0
    ll = int(len(r) / 2)
    v1 = r[:ll]
    v2 = r[ll:]
    return v1 + 1j * v2


def vectorize_complex_func(func: callable) -> np.ndarray:
    def vfunc(arr):
        out = func(double_real_to_complex(arr))
        return complex_to_double_real(out)

    return vfunc


def rebuild_matrices(arr: np.ndarray, nqm: int):

    sml = int(2 * (nqm**2))
    sms = int(2 * ((nqm - 1) ** 2))

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
    ) = rebuild_matrices(arr, nqm)

    H_eff_Nb = H_effective(gs_expectation_l, expectations_l, overlap_l)
    H_eff_Ns = H_effective(gs_expectation_s, expectations_s, overlap_s)

    return dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns)


def unscented_transform(
    func: callable,
    vals: np.ndarray,
    stds: np.ndarray,
    alpha: float = 1e-3,
    kappa: float = 0,
    beta: float = 2,
) -> tuple[np.ndarray, np.ndarray]:
    # brought to you by
    # The Unscented Kalman Filter for Nonlinear Estimation
    # Eric A. Wan and Rudolph van der Merwe
    # and my desire to end it all

    # anyway there is no covariance between the matrix variables. Hopefully

    test = func(vals)
    ll = test.shape
    if ll == ():
        ll = 1
    else:
        ll = ll[0]
    L = len(vals)
    lam = (alpha**2) * (L + kappa) - L

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

    for m in range(L):
        Xs[:, L + m + 1] = vals
        Xs[m, L + m + 1] -= stdsscaled[m]
        Ys[:, L + m + 1] = func(Xs[:, L + m + 1])

    ymean = Ys @ Wms

    Pmat = np.zeros((ll, ll), dtype=complex)
    for m in range(2 * L + 1):
        Pmat += Wcs[m] * (np.outer(Ys[:, m] - ymean, Ys[:, m] - ymean))

    if len(Pmat.shape) == 2:
        off = np.copy(Pmat)
        off -= np.diag(np.diag(off))
        print(f"Pmat has covariances, max abs off diag {np.max(np.abs(off))}")
        print(f"Max variance: {np.max(np.diag(Pmat))}")
        print("Ignoring covariances.")
        Pmat = np.diag(Pmat)

    ymean = ymean.flatten()
    Pmat = Pmat.flatten()

    assert np.all(ymean.real == ymean)
    assert np.all(Pmat.real == Pmat)

    return ymean.real, Pmat.real


def dispersion_1d_thermodynamic_ks_UT(
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

    assert np.all(arr.real == arr)
    means = []
    Pmats = []

    nqm = overlap_l.shape[0]

    vals = []

    for k in ks:

        def func(x):
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

    return (
        np.array(vals).flatten(),
        np.array(means).flatten(),
        np.array(Pmats).flatten(),
    )


def disp_from_res_dict(
    ks: np.ndarray,
    _d: dict,
    circ: Literal["vqe", "sweep"],
    improvement_needed: int = 1,
) -> dict:
    res = _d["results"][circ]
    disps = {}
    # tweak this to see how much better we need to get
    # 1 for actual values
    if improvement_needed != 1:
        print(f"Fictional scenario with std divided by {improvement_needed}")
    for be in res["dict_l"]["eld"]:
        if be != "system":
            overlap_l = res["dict_l"]["eld"][be]["values"]
            overlap_std_l = res["dict_l"]["eld"][be]["stds"] / improvement_needed
            expectations_l = res["dict_l"]["exd"][be]["values"]
            expectations_std_l = res["dict_l"]["exd"][be]["stds"] / improvement_needed
            gs_expectation_l = res["dict_l"]["gsd"][be]["energy"]
            gs_expectation_std_l = res["dict_l"]["gsd"][be]["std"] / improvement_needed

            overlap_s = res["dict_s"]["eld"][be]["values"]
            overlap_std_s = res["dict_s"]["eld"][be]["stds"] / improvement_needed
            expectations_s = res["dict_s"]["exd"][be]["values"]
            expectations_std_s = res["dict_s"]["exd"][be]["stds"] / improvement_needed
            gs_expectation_s = res["dict_s"]["gsd"][be]["energy"]
            gs_expectation_std_s = res["dict_s"]["gsd"][be]["std"] / improvement_needed

            vals, means, pmat = dispersion_1d_thermodynamic_ks_UT(
                ks,
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
            disps[be] = {"vals": vals, "means": means, "pmat": pmat}
    return disps
