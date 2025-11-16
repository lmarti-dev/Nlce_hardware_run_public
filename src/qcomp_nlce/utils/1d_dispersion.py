import numpy as np


def cluster_dispersion(N, H_effective, k):
    mat_el=[]
    for i in range(N):
        for j in range(N):
            # Calculate the phase factor exp(1im*k*(i-j)) and multiply with the matrix element
            func = H_effective[i, j] * np.exp(1j * k * (i - j))
            # Append the result to the "big" array
            mat_el.append(func)

    return sum(mat_el)


def dispersion_1d_thermodynamic(k, N, H_eff_Nb, H_eff_Ns):
    
    # Compute the difference between the sums of the "big" and "small" arrays
    func_w = (cluster_dispersion(N, H_eff_Nb, k) - cluster_dispersion(N-1, H_eff_Ns, k)).real
    
    # Return the final computed value
    return func_w

