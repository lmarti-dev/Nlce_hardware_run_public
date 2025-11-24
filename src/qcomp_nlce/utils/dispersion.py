import numpy as np


def cluster_dispersion(N, H_effective, k) -> float:
    mat_el = []
    for i in range(N):
        for j in range(N):

            # Calculate the phase factor exp(1im*k*(i-j)) and multiply with the matrix element
            # Here k is the momentum, and i, j are the site indices
            func = H_effective[i, j] @ np.exp(1j @ k @ (i - j))
            # Append the result to the "big" array
            mat_el.append(func)

    return np.sum(mat_el)


# This is the final function to be evaluated: see Eq. 51 in the paper

# N: number of spins in the largest cluster
# k= momentum, can taken anywhere between 0 to pi ( for 1D, the gap closing occurs at k=0, making it an interesting point to look at)
# H_effective_Nb : H_effective for bigger cluster of two (N spins)
# H_effective_Ns : H_effective for smaller cluster of two (N-1 spins)

def dispersion_1d_thermodynamic(k, N, H_eff_Nb, H_eff_Ns):

    # Compute the difference between the sums of the "big" and "small" arrays
    func_w = np.array(
        cluster_dispersion(N, H_eff_Nb, k) - cluster_dispersion(N - 1, H_eff_Ns, k)
    ).real

    # Return the final computed value
    return func_w
