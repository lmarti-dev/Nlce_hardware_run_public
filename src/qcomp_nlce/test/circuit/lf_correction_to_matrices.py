from src.qcomp_nlce.utils.workflow_utils import pick_model
from src.qcomp_nlce.utils import (
    get_tfim_sweep_circuit,
    sweep_time_from_steps,
    circuit_matrix_elements,
    hamming_weight_states,
    get_1qp_mat,
    get_ground_energy,
    dispersion_1d_thermodynamic,
    H_effective,
)
from src.qcomp_nlce.utils.results import pretty_print, pretty_print_diff
import numpy as np
from scipy.linalg import sqrtm, inv
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp


def print_lf_cdiff():
    n_qubits = 5
    J = 0.5

    hl = 0.1
    ladder = False
    n_steps = 20

    ham_j, ham_h = pick_model(n_qubits, J, hl, ladder)
    hamiltonian = ham_j + ham_h
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )

    hmat = hamiltonian.to_matrix()
    comp_states = hamming_weight_states(n_qubits, 1)
    overlap_naked = circuit_matrix_elements(circuit, comp_states)

    h_naked = get_1qp_mat(hamiltonian, False)


def get_heff(overlaps: np.ndarray, expectations: np.ndarray, e0: float) -> np.ndarray:
    aa_dag = overlaps @ overlaps.conj().T
    # matrix square root (not elementwise)
    sq_rt = sqrtm(aa_dag)
    T = overlaps.conj().T @ inv(sq_rt)

    # transform Hamiltonian into effective Hamiltonian; subtract GS energy
    n_qubits = len(overlaps)
    H_eff = T.conj().T @ expectations @ T - e0 * np.eye(n_qubits, dtype=complex)
    return H_eff


def expect_and_overlaps(hamiltonian: SparsePauliOp):
    n_qubits = hamiltonian.num_qubits

    hmat = hamiltonian.to_matrix()

    vals, vecs = np.linalg.eigh(hmat)

    e0 = np.min(vals)
    idx = np.array([2**j for j in range(n_qubits)], dtype=int)

    vecs_1qp = vecs[:, idx]
    gs = vecs[:, 0]

    vecs_h0_1qp = np.zeros((2**n_qubits, n_qubits))
    vecs_h0_1qp[idx, range(n_qubits)] = 1

    gs_h0 = np.zeros((2**n_qubits))
    gs_h0[0] = 1

    # 1 x 1
    gs_ov = gs_h0.conj().dot(gs)

    hcorr = np.zeros((n_qubits, n_qubits), dtype=complex)
    hnaked = np.zeros((n_qubits, n_qubits), dtype=complex)
    ocorr = np.zeros((n_qubits, n_qubits), dtype=complex)
    onaked = np.zeros((n_qubits, n_qubits), dtype=complex)

    for j in range(n_qubits):
        for k in range(n_qubits):
            hnaked[j, k] = vecs_1qp[:, j].T.conj() @ hmat @ vecs_1qp[:, k]

            cross1 = (
                vecs_1qp[:, j].T.conj()
                @ hmat
                @ gs
                * (gs_h0.dot(vecs_1qp[:, k]) / gs_ov)
            )
            cross2 = (
                gs.T.conj()
                @ hmat
                @ vecs_1qp[:, k]
                * (gs_h0.dot(vecs_1qp[:, j]) / gs_ov).conj()
            )

            sq2 = (gs.conj().T @ hmat @ gs) * (
                (gs_h0 @ vecs_1qp[:, k])
                * (gs_h0 @ vecs_1qp[:, j])
                / (np.abs(gs_ov)) ** 2
            )

            hcorr[j, k] = hnaked[j, k] - cross1 - cross2 + sq2
            onaked[j, k] = vecs_h0_1qp[:, j].T.conj() @ vecs_1qp[:, k]

            ocorr[j, k] = onaked[j, k] - vecs_h0_1qp[:, j].T.conj() @ gs * (
                gs_h0.dot(vecs_1qp[:, k]) / gs_ov
            )

    # print("expects")

    # pretty_print(hnaked)
    # pretty_print(hcorr)
    # pretty_print_diff(hnaked, hcorr)

    # print("overlaps")

    # pretty_print(onaked)
    # pretty_print(ocorr)
    # pretty_print_diff(onaked, ocorr)

    return hnaked, hcorr, onaked, ocorr, e0


def plot_disp_corr(n_qubits: int, J: float, hl: float, ladder: bool = False):
    ks = np.linspace(-np.pi, np.pi, 21)
    ham_j, ham_h = pick_model(n_qubits, J, hl, ladder)
    hamiltonian_l = ham_j + ham_h
    hnaked_l, hcorr_l, onaked_l, ocorr_l, e0_l = expect_and_overlaps(hamiltonian_l)

    ham_j, ham_h = pick_model(n_qubits - 1, J, hl, ladder)
    hamiltonian_s = ham_j + ham_h
    hnaked_s, hcorr_s, onaked_s, ocorr_s, e0_s = expect_and_overlaps(hamiltonian_s)

    heff_naked_l = H_effective(e0_l, hnaked_l, onaked_l)
    heff_naked_s = H_effective(e0_s, hnaked_s, onaked_s)

    heff_corr_l = H_effective(e0_l, hcorr_l, ocorr_l)
    heff_corr_s = H_effective(e0_s, hcorr_s, ocorr_s)

    disp_naked = [
        dispersion_1d_thermodynamic(k, heff_naked_l, heff_naked_s) for k in ks
    ]
    disp_corr = [dispersion_1d_thermodynamic(k, heff_corr_l, heff_corr_s) for k in ks]

    fig, ax = plt.subplots()

    ax.plot(ks, disp_naked, label="uncorrected")
    ax.plot(ks, disp_corr, label="corrected")

    plt.show()


if __name__ == "__main__":
    n_qubits = 5
    J = 0.5

    hl = 0.1
    ladder = False

    plot_disp_corr(n_qubits, J, hl, ladder)
