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


def _ladder_coordinates(n_rungs: int) -> list[tuple[int, int]]:
    """Return (x, y) coordinates for each qubit in a ladder.

    Qubit ordering matches TFIM ladder Hamiltonian:
        indices 0..n_rungs-1         -> chain 1: (rung, leg=0)
        indices n_rungs..2*n_rungs-1 -> chain 2: (rung, leg=1)

    Returns list of (x, y) tuples where x = rung index, y = leg index.
    """
    coords = []
    for leg in range(2):
        for rung in range(n_rungs):
            coords.append((rung, leg))
    return coords


def cluster_dispersion_ladder(
    n_rungs: int, H_effective: np.ndarray, kx: float, ky: float
) -> float:
    """Fourier transform H_eff for a ladder cluster at momentum (kx, ky).

    Following the 2D convention from the Julia code:
      - For n_x > n_y (n_rungs > 2): sum both orientations
            cos(hopx*kx + hopy*ky) + cos(hopx*ky + hopy*kx)
      - For n_x == n_y (2x2): single orientation
            cos(hopx*kx + hopy*ky)

    Here n_x = n_rungs, n_y = 2.
    """
    n_sites = 2 * n_rungs
    coords = _ladder_coordinates(n_rungs)
    n_x = n_rungs
    n_y = 2

    total = 0.0
    for i in range(n_sites):
        for j in range(n_sites):
            hopx = coords[j][0] - coords[i][0]
            hopy = coords[j][1] - coords[i][1]

            if n_x > n_y:
                fourier = H_effective[i, j] * np.cos(hopx * kx + hopy * ky)
                fourier += H_effective[i, j] * np.cos(hopx * ky + hopy * kx)
            else:  # n_x == n_y (2x2 case)
                fourier = H_effective[i, j] * np.cos(hopx * kx + hopy * ky)

            total += fourier

    return total


def dispersion_ladder_thermodynamic(
    kx: float, ky: float, H_eff_Nb: np.ndarray, H_eff_Ns: np.ndarray
) -> float:
    """NLCE dispersion for ladder: E(Nx2) - E((N-1)x2).

    Returns the real-valued NLCE-subtracted dispersion at momentum (kx, ky).
    """
    Nb = H_eff_Nb.shape[0] // 2  # number of rungs in bigger cluster
    Ns = H_eff_Ns.shape[0] // 2  # number of rungs in smaller cluster
    return (
        cluster_dispersion_ladder(Nb, H_eff_Nb, kx, ky)
        - cluster_dispersion_ladder(Ns, H_eff_Ns, kx, ky)
    )


def exact_dispersion_1qp_1d(j: float, k: np.ndarray):
    return 2 * np.sqrt(1 + j**2 - 2 * j * np.cos(k))


def heff_n_expectation_values_1d(n_qubits: int, hl: bool = False) -> int:

    total = 0

    for q in (n_qubits - 1, n_qubits):

        # 2 per overlap element
        overlap = 2 * q**2
        # 1 per diagonal, 4 per nondiagonal element
        # for a 1d ising model, one needs 3 circuits to measure the Hamiltonian:
        # 1 for sum Z, 1 for odd XX, and 1 for even XX.
        # so 3 totl
        # if Hl, there's one more for the X field.
        if hl:
            ff = 4
        else:
            ff = 3
        expectations = ff * (q + 4 * sum(range(1, q)))
        gs = 1
        total += sum([overlap, expectations, gs])
    return total


def heff_n_expectation_values_ladder(n_qubits: int) -> int:
    total = 0
    for q in (n_qubits - 2, n_qubits):
        total += 2 * q**2 + 4 * (q + 4 * sum(range(1, q))) + 1
    return total


def heff_n_expectation_values_2d(n_qubits: int) -> int:
    total = 0

    # in the 2d case, for the hamiltonian, we have
    # 1 for sum Z 2 XX horizonally, and 2 XX vertically
    # so 5 total

    # 1D chains: only two largest
    for m in (n_qubits - 1, n_qubits):
        q = m
        total += 2 * q**2 + 3 * (q + 4 * sum(range(1, q))) + 1

    # 2D rectangles: m >= n >= 2, boundary only
    for m in range(2, n_qubits + 1):
        for n in range(2, m + 1):
            if m * n <= n_qubits and (m + 2) * (n + 2) > n_qubits:
                q = m * n
                total += 2 * q**2 + 5 * (q + 4 * sum(range(1, q))) + 1

    return total


# def heff_n_expectation_values_1d(n_qubits: int) -> int:
#     ...
#     return total


if __name__ == "__main__":
    ...
