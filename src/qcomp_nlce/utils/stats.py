import numpy as np
from uncertainties import unumpy, ufloat, Variable


def prod_std(stds: np.ndarray, vals: np.ndarray) -> float:
    stds = np.array(stds)
    vals = np.array(vals)
    return np.sqrt(np.sum((stds**2) / (vals**2)))


def sum_std(stds: np.ndarray) -> float:
    stds = np.array(stds)
    return np.sqrt(np.sum((stds**2)))


def u_inv(m: unumpy.matrix) -> unumpy.matrix:
    # 1/(1-x) = 1 + x + x**2 + x**3 + ...
    eye = unumpy.matrix(np.eye(m.shape[0]))
    k = eye - m
    return eye + k + k @ k


def u_isqrtm(m: unumpy.matrix):
    # (1+x)**(-1/2) = 1 - 1/2 x + 3/8 x**2
    # for ||1-m|| < 1
    eye = unumpy.matrix(np.eye(m.shape[0]))
    k = eye - m
    return eye - 0.5 * k + (3 / 8) * k @ k


def u_dag(m: unumpy.matrix):
    m_vals = unumpy.nominal_values(m)
    m_stds = unumpy.std_devs(m)
    return unumpy.umatrix(np.conj(m_vals).T, m_stds.T)


def u_diag(n: int, ede: Variable) -> unumpy.matrix:
    e = ede.nominal_value
    de = ede.std_dev
    v = np.diag(np.repeat(e, n))
    dv = np.diag(np.repeat(de, n))
    return unumpy.umatrix(v, dv)


def u_dispersion_1d_thermodynamic(
    k: float, H_eff_Nb: np.ndarray, H_eff_Ns: np.ndarray
) -> float:

    # Compute the difference between the sums of the "big" and "small" arrays
    N = H_eff_Nb.shape[0]
    func_w = np.array(
        u_cluster_dispersion(N, H_eff_Nb, k) - u_cluster_dispersion(N - 1, H_eff_Ns, k)
    ).real

    # Return the final computed value
    return func_w


def u_dispersion_1d_thermodynamic_ks(
    ks: np.ndarray, H_eff_Nb: np.ndarray, H_eff_Ns: np.ndarray
) -> np.ndarray:
    out = np.zeros((2, len(ks)))
    for ind, k in enumerate(ks):
        g = u_dispersion_1d_thermodynamic(k, H_eff_Nb, H_eff_Ns)
        out[0][ind] = unumpy.nominal_values(g)
        out[1][ind] = unumpy.std_devs(g)

    return out


def u_cluster_dispersion(N: int, H_effective: np.ndarray, k: float) -> float:
    mat_el = ufloat(0, 0)
    for i in range(N):
        for j in range(N):

            # Calculate the phase factor exp(1im*k*(i-j)) and multiply with the matrix element
            # Here k is the momentum, and i, j are the site indices
            func = H_effective[i, j] * ufloat(np.exp(1j * k * (i - j)), 0)
            # Append the result to the "big" array
            mat_el += func

    return mat_el


def u_H_effective(
    gs_expectation: Variable,
    one_qp_expectation: unumpy.matrix,
    overlap: unumpy.matrix,
) -> unumpy.matrix:
    # implements the uncertainties version of it
    # one qp expectation: <psi|U* H U |psi>
    # overlap: <psi|U|psi>
    #  gs expectation is ground state energy

    # Get dimension from one_qp_expectation matrix
    N = one_qp_expectation.shape[0]
    eye = unumpy.matrix(np.eye(one_qp_expectation.shape[0]))

    # Compute transformation matrix: B = (overlap @ overlap†)^(-1/2) @ overlap
    inv_sqrt = u_isqrtm(overlap @ u_dag(overlap))
    B = u_dag(overlap) @ inv_sqrt
    # Apply PCAT transformation (Eq. 47)
    u_H_effective = u_dag(B) @ one_qp_expectation @ B - eye * u_diag(N, gs_expectation)

    return u_H_effective


def u_heff_from_numpy(
    overlap_val: np.ndarray,
    overlap_std: np.ndarray,
    expectations_vals: np.ndarray,
    expectations_std: np.ndarray,
    gs_expectation: float,
    gs_expectation_std: float,
):
    u_expec = unumpy.umatrix(expectations_vals, expectations_std)
    u_overl = unumpy.umatrix(overlap_val, overlap_std)
    u_gs = ufloat(gs_expectation, gs_expectation_std)
    h_eff = u_H_effective(u_gs, u_expec, u_overl)
    return h_eff
