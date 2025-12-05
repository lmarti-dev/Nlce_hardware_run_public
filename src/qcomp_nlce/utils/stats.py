import numpy as np


def prod_std(stds: np.ndarray, vals: np.ndarray) -> float:
    stds = np.array(stds)
    vals = np.array(vals)
    return np.sqrt(np.sum((stds**2) / (vals**2)))


def sum_std(stds: np.ndarray) -> float:
    stds = np.array(stds)
    return np.sqrt(np.sum((stds**2)))
