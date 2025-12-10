import numpy as np


def prod_std(stds: np.ndarray, vals: np.ndarray) -> float:
    stds = np.array(stds)
    vals = np.array(vals)
    return np.sqrt(np.sum((stds**2) / (vals**2)))


def sum_std(stds: np.ndarray, axis: int = 0) -> float:
    stds = np.array(stds)
    return np.sqrt(np.sum(stds**2, axis=axis))


def mean_std(stds: np.ndarray, axis: int = 0) -> float:
    return sum_std(stds, axis=axis) / len(stds)
