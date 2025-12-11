import numpy as np
from qcomp_nlce.utils.dispersion import dispersion_1d_thermodynamic
from qcomp_nlce.utils.constants import ALGORITHMS_ALIAS
from qcomp_nlce.utils.effective_hamiltonian import H_effective
from qcomp_nlce.utils.save import load_result_matrices
from qcomp_nlce.utils.hardware import get_provider_version
from pathlib import Path


def prod_std(stds: np.ndarray, vals: np.ndarray) -> float:
    stds = np.array(stds)
    vals = np.array(vals)
    return np.sqrt(np.sum((stds**2) / (vals**2)))


def sum_std(stds: np.ndarray, axis: int = 0) -> float:
    stds = np.array(stds)
    return np.sqrt(np.sum(stds**2, axis=axis))


def mean_std(stds: np.ndarray, axis: int = 0) -> float:
    return sum_std(stds, axis=axis) / len(stds)


def complex_normal(loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.random.normal(loc=loc.real, scale=scale.real)
    y = np.random.normal(loc=loc.imag, scale=scale.imag)
    return x + 1j * y


def heff_mc_sampler_from_file(fpath: Path = None) -> callable:
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
    ) = load_result_matrices(fpath)

    get_heff_mc_sampler(
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


def get_heff_mc_sampler(
    expect_l: np.ndarray,
    expect_std_l: np.ndarray,
    overlap_l: np.ndarray,
    overlap_std_l: np.ndarray,
    gs_l: np.ndarray,
    gs_std_l: np.ndarray,
    expect_s: np.ndarray,
    expect_std_s: np.ndarray,
    overlap_s: np.ndarray,
    overlap_std_s: np.ndarray,
    gs_s: np.ndarray,
    gs_std_s: np.ndarray,
) -> callable:

    def sampler(ks, fac: float = 1.0):
        sa_expect_l = complex_normal(loc=expect_l, scale=expect_std_l * fac)
        sa_overlap_l = complex_normal(loc=overlap_l, scale=overlap_std_l * fac)
        sa_gs_l = complex_normal(loc=gs_l, scale=gs_std_l * fac)

        sa_expect_s = complex_normal(loc=expect_s, scale=expect_std_s * fac)
        sa_overlap_s = complex_normal(loc=overlap_s, scale=overlap_std_s * fac)
        sa_gs_s = complex_normal(loc=gs_s, scale=gs_std_s * fac)

        H_eff_Nb = H_effective(sa_gs_l, sa_expect_l, sa_overlap_l)
        H_eff_Ns = H_effective(sa_gs_s, sa_expect_s, sa_overlap_s)

        return [dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns) for k in ks]

    return sampler


def mc_disp_from_res_dict(
    ks: np.ndarray, _d: dict, circ: ALGORITHMS_ALIAS, n_samples: int = 10000
) -> dict:
    res = _d["results"][circ]
    disps = {}

    fac = 1
    # if the provider is v1
    # the std is the std of the observable
    # and not the error on the mean
    # which is what we want to propagate
    # because we're not looking at the true distribution
    # but rather the distribution of the estimator
    if get_provider_version(_d["inputs"]["provider_name"]) == "v1":
        n_shots = _d["inputs"]["n_shots"]
        fac = 1 / np.sqrt(n_shots)

    for be in res["dict_l"]["eld"]:
        if be != "system":
            overlap_l = res["dict_l"]["eld"][be]["values"]
            overlap_std_l = res["dict_l"]["eld"][be]["stds"] * fac
            expectations_l = res["dict_l"]["exd"][be]["values"]
            expectations_std_l = res["dict_l"]["exd"][be]["stds"] * fac
            gs_expectation_l = res["dict_l"]["gsd"][be]["energy"]
            gs_expectation_std_l = res["dict_l"]["gsd"][be]["std"]

            overlap_s = res["dict_s"]["eld"][be]["values"]
            overlap_std_s = res["dict_s"]["eld"][be]["stds"] * fac
            expectations_s = res["dict_s"]["exd"][be]["values"]
            expectations_std_s = res["dict_s"]["exd"][be]["stds"] * fac
            gs_expectation_s = res["dict_s"]["gsd"][be]["energy"]
            gs_expectation_std_s = res["dict_s"]["gsd"][be]["std"]

            sampler = get_heff_mc_sampler(
                expectations_l,
                expectations_std_l,
                overlap_l,
                overlap_std_l,
                gs_expectation_l,
                gs_expectation_std_l,
                expectations_s,
                expectations_std_s,
                overlap_s,
                overlap_std_s,
                gs_expectation_s,
                gs_expectation_std_s,
            )

            sampled = np.zeros((len(ks), n_samples))
            for n in range(n_samples):
                print(f"{n:0>6n}/{n_samples}", end="\r")
                sampled[:, n] = sampler(ks)

            mc_mean = np.mean(sampled, axis=1)
            mc_std = np.std(sampled, axis=1)

            disps[be] = {
                "mean": mc_mean,
                "std": mc_std,
                "n_samples": n_samples,
            }
    return disps
