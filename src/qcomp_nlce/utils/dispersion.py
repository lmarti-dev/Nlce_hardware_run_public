import numpy as np


def cluster_dispersion(N: int, H_effective: np.ndarray, k: float) -> float:
    mat_el = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):

            # Calculate the phase factor exp(1im*k*(i-j)) and multiply with the matrix element
            # Here k is the momentum, and i, j are the site indices
            func = H_effective[i, j] * np.exp(1j * k * (i - j))
            # Append the result to the "big" array
            mat_el[i, j] = func

    return np.sum(mat_el)


# This is the final function to be evaluated: see Eq. 51 in the paper


# N: number of spins in the largest cluster
# k= momentum, can taken anywhere between 0 to pi ( for 1D, the gap closing occurs at k=0, making it an interesting point to look at)
# H_effective_Nb : H_effective for bigger cluster of two (N spins)
# H_effective_Ns : H_effective for smaller cluster of two (N-1 spins)


def dispersion_1d_thermodynamic(
    k: float, H_eff_Nb: np.ndarray, H_eff_Ns: np.ndarray
) -> float:

    # Compute the difference between the sums of the "big" and "small" arrays
    Nb = H_eff_Nb.shape[0]
    Ns = H_eff_Ns.shape[0]
    func_w = np.array(
        cluster_dispersion(Nb, H_eff_Nb, k) - cluster_dispersion(Ns, H_eff_Ns, k)
    ).real

    # Return the final computed value
    return func_w


def _ladder_momentum_projectors(n_rungs: int, kx: float) -> tuple[np.ndarray, np.ndarray]:
    
    n_sites = 2 * n_rungs
    v_leg0 = np.zeros(n_sites, dtype=complex)
    v_leg1 = np.zeros(n_sites, dtype=complex)

    for r in range(n_rungs):
        phase = np.exp(1j * kx * r)
        v_leg0[r] = phase              # leg 0: sites 0..n_rungs-1
        v_leg1[n_rungs + r] = phase    # leg 1: sites n_rungs..2*n_rungs-1

    return v_leg0, v_leg1


def project_ladder_to_k_space(
    H_eff: np.ndarray, n_rungs: int, kx: float
) -> np.ndarray:

    v0, v1 = _ladder_momentum_projectors(n_rungs, kx)
    H_k = np.zeros((2, 2), dtype=complex)
    H_k[0, 0] = v0.conj() @ H_eff @ v0
    H_k[0, 1] = v0.conj() @ H_eff @ v1
    H_k[1, 0] = v1.conj() @ H_eff @ v0
    H_k[1, 1] = v1.conj() @ H_eff @ v1
    return H_k


def dispersion_ladder_thermodynamic(
    k: float, H_eff_Nb: np.ndarray, H_eff_Ns: np.ndarray
) -> np.ndarray:

    Nb = H_eff_Nb.shape[0] // 2  # number of rungs in bigger cluster
    Ns = H_eff_Ns.shape[0] // 2  # number of rungs in smaller cluster

    H_k_big = project_ladder_to_k_space(H_eff_Nb, Nb, k)
    H_k_small = project_ladder_to_k_space(H_eff_Ns, Ns, k)

    H_k_inf = H_k_big - H_k_small
    evals = np.sort(np.real(np.linalg.eigvals(H_k_inf)))
    return evals


def exact_dispersion_1qp_1d(j: float, k: np.ndarray):
    return 2 * np.sqrt(1 + j**2 - 2 * j * np.cos(k))


def heff_n_expectation_values_1d(n_qubits: int, ff: int = 2) -> int:

    total = 0

    for q in (n_qubits - 1, n_qubits):

        # 2 per overlap element
        overlap = 2 * q**2
        # 1 per diagonal, 4 per nondiagonal element
        # for a 1d ising model, one needs 3 circuits to measure the Hamiltonian:
        # 1 for sum Z, 1 for XX.
        # so ff=2 total
        # if Hl, there's one more for the X field. so ff=3
        expectations = ff * (q + 4 * sum(range(1, q)))
        gs = 1
        total += sum([overlap, expectations, gs])
    return total


def heff_n_expectation_values_ladder(n_qubits: int, ff: int = 2) -> int:
    total = 0
    for q in (n_qubits - 2, n_qubits):
        total += 2 * q**2 + ff * (q + 4 * sum(range(1, q))) + 1
    return total


def heff_n_expectation_values_2d(n_qubits: int, ff: int = 2) -> int:
    total = 0

    # in the 2d case, for the hamiltonian, we have
    # 1 for sum Z 2 XX horizonally, and 2 XX vertically
    # so 5 total

    # 1D chains: only two largest
    for m in (n_qubits - 1, n_qubits):
        q = m
        total += 2 * q**2 + ff * (q + 4 * sum(range(1, q))) + 1

    # 2D rectangles: m >= n >= 2, boundary only
    for m in range(2, n_qubits + 1):
        for n in range(2, m + 1):
            if m * n <= n_qubits and (m + 2) * (n + 2) > n_qubits:
                q = m * n
                total += 2 * q**2 + ff * (q + 4 * sum(range(1, q))) + 1

    return total


if __name__ == "__main__":
    ...
