from qcomp_nlce.utils.unscented_transform import (
    unscented_transform,
    vectorize_complex_func,
    double_real_to_complex,
    complex_to_double_real,
)
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 2 * x + np.cos(x) + 0.033 * x**2 + np.linalg.norm(x) ** (0.05)


def mcf(func: callable, n_samples: int, mus: np.ndarray, sigmas: np.ndarray):
    assert len(mus) == len(sigmas)

    shape = (len(mus), n_samples)
    inputs = np.zeros(shape)
    for iii, (x, y) in enumerate(zip(mus, sigmas)):
        inputs[iii, :] = np.random.normal(loc=x, scale=y, size=n_samples)
    out = np.zeros(shape, dtype=complex)
    for jj in range(n_samples):
        print(f"{jj:0>10n}", end="\r")
        y = func(inputs[:, jj])
        out[:, jj] = y
    return np.mean(out, axis=1).real, np.std(out, axis=1).real, out


def test_UT_with_normal():
    n_samples = 100
    mu_x = 0
    mu_y = 1

    sigma_x = 1
    sigma_y = 2

    x = np.random.normal(mu_x, sigma_x, size=n_samples)
    y = np.random.normal(mu_y, sigma_y, size=n_samples)
    z_mean = np.mean(x + 1j * y)

    std = np.std(x) + 1j * np.std(y)

    ymean, pmat = unscented_transform(
        vectorize_complex_func(func),
        complex_to_double_real(z_mean),
        complex_to_double_real(std),
    )

    mcf_mean, mcf_std, mc_data = mcf(
        vectorize_complex_func(func),
        int(1e3 * n_samples),
        mus=[mu_x, mu_y],
        sigmas=[sigma_x, sigma_y],
    )

    ystd = pmat ** (1 / 2)

    fmean = np.mean(func(x + 1j * y))
    fstd = np.std(func(x + 1j * y).real) + 1j * np.std(func(x + 1j * y).imag)

    fig, ax = plt.subplots()
    ax.plot(func(x + 1j * y).real, func(x + 1j * y).imag, ".", label="data")
    ax.plot(mc_data[0, :], mc_data[1, :], "g.", alpha=0.5, label="mc data")
    ax.errorbar(fmean.real, fmean.imag, fstd.real, fstd.imag, "rx", label="fmean")
    ax.errorbar(mcf_mean[0], mcf_mean[1], mcf_std[0], mcf_std[1], "bd", label="mc")
    ax.errorbar(ymean[0], ymean[1], ystd[0], ystd[1])

    ax.legend()

    print(f"True mean real {fmean.real:.5f} f(std) real {fstd.real:.5f} ")
    print(f"True mean imag {fmean.imag:.5f} f(std) imag {fstd.imag:.5f} ")
    print(f"Estimated mean real {ymean[0]:.5f} std real {ystd[0]:.5f}")
    print(f"Estimated mean imag {ymean[1]:.5f} std imag {ystd[1]:.5f}")
    print(f"MCF mean real: {mcf_mean[0]:.5f}, std: {mcf_std[0]:.5f}")
    print(f"MCF mean imag: {mcf_mean[1]:.5f}, std: {mcf_std[1]:.5f}")

    plt.show()


if __name__ == "__main__":
    test_UT_with_normal()
