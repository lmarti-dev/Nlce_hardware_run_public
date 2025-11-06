import numpy as np
import scipy.special as sp


# The transverse field Ising model in 1D is exactly solvable
# Calculates the exact energy for a one-dimensional transverse-field Ising chain in the thermodynamic limit
def pfeuty_energy(J) -> float:
    h = 1
    m = (4 * h * J) / ((h + J) ** 2)
    pfeuty = -(4 * (h + J)) * (sp.ellipe(m)) * (1 / (2 * np.pi))
    return pfeuty

# NLCE eembedding for 1D TFIM which needs only two clusters of sizes N and N-1
def nlce_energy(A, B):
    energy = []
    for i, j in zip(A, B):
        diff = i - j
        nlce = diff
        energy.append(nlce)
    return energy
