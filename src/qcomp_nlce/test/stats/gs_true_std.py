from src.qcomp_nlce.utils import (
    TFIM,
    get_ground_state,
    from_bitstring,
    subspace_diagonalize,
    get_unique_bitstrings,
)
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.result import Counts
import numpy as np


def ham_std(ham: np.ndarray, state: np.ndarray) -> float:
    v1 = (state.conj().T @ ham).dot(ham @ state)
    v2 = (state.conj().T @ ham).dot(state) ** 2
    return v1 - v2


def resample_counts(counts: dict, n_shots: int):
    k = list(counts.keys())
    picks = np.random.choice(k, n_shots, replace=True)
    out = {}
    for p in picks:
        if p in out:
            out[p] += 1
        else:
            out[p] = 1
    return out


def bootstrap_std(ham: SparsePauliOp, counts: dict, n_shots: int, n_straps: int):
    es = []
    for strap in range(n_straps):
        new_counts = resample_counts(counts, n_shots)
        ubits = get_unique_bitstrings(new_counts)
        e = subspace_diagonalize(ubits, ham)
        es.append(e)
    return np.mean(es), np.std(es)


def build_from_counts(counts: dict, n_qubits: int, n_shots: int) -> np.ndarray:
    state = np.zeros((2**n_qubits))

    for c in counts:
        jj = from_bitstring(c, n_qubits, True)
        state[jj] = counts[c] / n_shots
    return state


def subspace_shots(n_shots: float, J):
    n_qubits = 5
    ham = TFIM(n_qubits, 1, J)
    hammat = ham.to_matrix()
    gs = get_ground_state(ham)
    sv = Statevector(gs)
    counts: Counts = sv.sample_counts(n_shots)
    app_state = build_from_counts(counts, n_qubits, n_shots)
    v_noisy = ham_std(hammat, app_state)
    v_true = ham_std(hammat, gs.T[0])
    print(f"true (0): {v_true:.5f}, noisy: {v_noisy:.5f}")


def default_stds():
    for J in (0.3, 0.5, 0.8, 1):
        subspace_shots(2000, J)


def bootstrap_std_tfim(n_shots: int, J: float):
    n_qubits = 5
    ham = TFIM(n_qubits, 1, J)
    gs = get_ground_state(ham)
    sv = Statevector(gs)
    counts: Counts = sv.sample_counts(n_shots)
    emean, estd = bootstrap_std(ham, counts, n_shots, 1000)
    print(f"emean: {emean:.5f}, estd: {estd:.5f}")


def mc_std_tfim(n_shots: int, J: float):
    n_mc_samples = 10000
    n_qubits = 5
    ham = TFIM(n_qubits, 1, J)
    gs = get_ground_state(ham)
    sv = Statevector(gs)
    es = []
    for mc in range(n_mc_samples):
        print(f"{mc / n_mc_samples}", end="\r")
        counts: Counts = sv.sample_counts(n_shots)
        ubits = get_unique_bitstrings(counts)
        e = subspace_diagonalize(ubits, ham)
        es.append(e)
    emean, estd = np.mean(es), np.std(es)
    print(f"J/h={J}: emean: {emean:.5f}, estd: {np.sqrt(n_shots) * estd:.5f}")


if __name__ == "__main__":
    for J in (0.3, 0.5, 0.8, 1):
        mc_std_tfim(2000, J)
