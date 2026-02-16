import numpy as np


def norm(alpha, n_qubits):
    return np.sqrt(np.sum([np.exp(-2 * alpha * k) for k in range(2**n_qubits)]))


alpha = 0.1

print(norm(alpha, 10))
