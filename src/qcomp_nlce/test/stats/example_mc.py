import numpy as np


def g(x: np.ndarray) -> float:
    return "very complicated function of x"


def montecarlo(loc: np.ndarray, sigma: np.ndarray, n_samples: int):
    sampled = np.zeros((n_samples,))

    for n in range(n_samples):
        x = np.random.normal(loc=loc, sigma=sigma)
        sampled[n] = g(x)

    return np.mean(sampled, axis=1), np.std(sampled, axis=1)


x_experimental_mean = (1.23, 1.54, 0.34)
x_experimental_std = (0.04, 0.02, 0.05)

mc_mean, mc_std = montecarlo(x_experimental_mean, x_experimental_std, 1000)
