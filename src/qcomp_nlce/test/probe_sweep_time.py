from qcomp_nlce.utils import (
    exact_effective_hamiltonian,
    TFIM,
    get_min_gap,
    get_tfim_ratio,
    get_tfim_sweep_circuit,
    hamming_weight_states,
    circuit_matrix_elements,
    H_effective,
    circ_ham_expectation_values,
    trace_dist,
    dispersion_1d_thermodynamic,
    from_bitstring,
    exact_dispersion_1qp_1d,
    H_effective_ed,
)


from fauplotstyle import style
from data_manager import ExperimentDataManager

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import matplotlib.pyplot as plt
import numpy as np


def circuit_energy(hamiltonian: SparsePauliOp, circuit: QuantumCircuit):
    sv = Statevector(circuit)
    return sv.expectation_value(hamiltonian)


def get_heff_sweep(n_qubits: int, mult: int, J: float):
    ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = mult / (min_gap**2)
    n_steps = int(mult * n_qubits * hamiltonian.num_qubits)

    # ensure at least two steps
    n_steps = max(n_steps, 2)

    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )

    print(
        f"{hamiltonian.num_qubits} qubits, J/h {get_tfim_ratio(hamiltonian)} mult: {mult}"
    )
    print(f"sweep depth: {circuit.depth()}")

    comp_states = hamming_weight_states(n_qubits, 1)
    overlap = circuit_matrix_elements(circuit, comp_states)
    one_qp_expectations = circ_ham_expectation_values(hamiltonian, circuit, comp_states)
    gs_expectation = circuit_energy(hamiltonian, circuit)

    heff_sweep = H_effective(gs_expectation, one_qp_expectations, overlap)

    return heff_sweep


def sweep_dispersion(n_qubits_max: int, mult: int, J: float, k: float = 0.0):
    heff_b = get_heff_sweep(n_qubits_max, mult, J)
    heff_s = get_heff_sweep(n_qubits_max - 1, mult, J)

    return dispersion_1d_thermodynamic(k, n_qubits_max, heff_b, heff_s)


def exact_dispersion(n_qubits_max: int, J: float, k: float = 0.0):
    ham_l = TFIM(n_qubits_max, 1, J, periodic=False)
    ham_s = TFIM(n_qubits_max - 1, 1, J, periodic=False)
    heff_exact_l = exact_effective_hamiltonian(ham_l)
    heff_exact_s = exact_effective_hamiltonian(ham_s)

    return dispersion_1d_thermodynamic(k, n_qubits_max, heff_exact_l, heff_exact_s)


def get_overlaps(mult: int, n_qubits: int, J: float):
    ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = 1 / (min_gap**2)
    n_steps = 10 * hamiltonian.num_qubits

    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=(mult * total_time),
        n_steps=int(mult * n_steps),
    )
    comp_states = hamming_weight_states(n_qubits, 1)
    overlap_sweep = circuit_matrix_elements(circuit, comp_states)

    _, eig_states = np.linalg.eigh(hamiltonian.to_matrix(False))
    sl = [
        from_bitstring(state, n_qubits=n_qubits, right_to_left=True)
        for state in comp_states
    ]
    el = list(range(1, n_qubits + 1))
    overlap_exact = eig_states[np.ix_(sl, el)]

    return overlap_sweep, overlap_exact


def probe_dispersion_diff_scan_k(n_qubits_max):

    J = 1.0

    mults = np.linspace(0.1, 1, 5)
    ks = np.linspace(0, np.pi, 21)

    sweep_disps = np.zeros((len(ks), len(mults) + 1))

    ham_l = TFIM(n_qubits_max, 1, J, periodic=False)
    ham_s = TFIM(n_qubits_max - 1, 1, J, periodic=False)

    heff_exact_l = H_effective_ed(ham_l)
    heff_exact_s = H_effective_ed(ham_s)

    def exact_dd(k: np.ndarray):
        return [
            dispersion_1d_thermodynamic(_k, n_qubits_max, heff_exact_l, heff_exact_s)
            for _k in k
        ]

    exact_disps = exact_dd(ks)
    sweep_disps[:, 0] = [identity_dispersion(n_qubits_max, J, k) for k in ks]

    for ind_m, mult in enumerate(mults):

        heff_b = get_heff_sweep(n_qubits_max, mult, J)
        heff_s = get_heff_sweep(n_qubits_max - 1, mult, J)

        def sweep_dd(k: np.ndarray):
            return [
                dispersion_1d_thermodynamic(_k, n_qubits_max, heff_b, heff_s)
                for _k in k
            ]

        sweep_disps[:, ind_m + 1] = sweep_dd(ks)

    fig, ax = plt.subplots()
    for m in range(sweep_disps.shape[1]):
        ax.plot(ks, sweep_disps[:, m], label=f"m={[0,*mults][m]}")
    ax.plot(ks, exact_disps, label="ED")
    ax.plot(ks, exact_dispersion_1qp_1d(J, ks), label="Exact")
    ax.legend()

    return fig, {
        "mults": mults,
        "j": J,
        "ks": ks,
        "sweep_disps": sweep_disps,
        "exact_disps": exact_disps,
        "n_qubits": n_qubits_max,
    }


def probe_dispersion_diff_scan_j(n_qubits_max):

    mults = np.linspace(1, 8, 3)
    js = np.linspace(0.6, 1.0, 10)
    sweep_disps = np.zeros((len(mults) + 1, len(js)))
    exact_disps = np.zeros((len(js),))

    for ind_j, j in enumerate(js):
        ii = identity_dispersion(n_qubits_max, j)
        sweep_disps[0, ind_j] = ii
        y = exact_dispersion(n_qubits_max, j)
        exact_disps[ind_j] = y
        for ind_m, mult in enumerate(mults):
            x = sweep_dispersion(n_qubits_max, mult, j)
            sweep_disps[ind_m + 1, ind_j] = x

    fig, ax = plt.subplots()
    for m in range(sweep_disps.shape[0]):
        ax.plot(js, sweep_disps[m, :], label=f"m={[0,*mults][m]}")
    ax.plot(js, exact_disps, label="ED")
    ax.legend()

    return fig, {
        "mults": mults,
        "js": js,
        "sweep_disps": sweep_disps,
        "exact_disps": exact_disps,
        "n_qubits": n_qubits_max,
    }


def identity_heff(n_qubits: int, J: float):
    hamiltonian = TFIM(n_qubits, 1, J, periodic=False)
    comp_states = hamming_weight_states(n_qubits, 1)
    one_qp_expectations = circ_ham_expectation_values(
        hamiltonian, QuantumCircuit(hamiltonian.num_qubits), comp_states
    )
    overlap = np.eye(n_qubits)
    heff = H_effective(-n_qubits, one_qp_expectations, overlap)
    return heff


def identity_dispersion(n_qubits_max: int, J: float, k: float = 0.0):
    heff_l = identity_heff(n_qubits_max, J)
    heff_s = identity_heff(n_qubits_max - 1, J)
    return dispersion_1d_thermodynamic(k, n_qubits_max, heff_l, heff_s)


def probe_overlaps():

    for mult in np.linspace(1e-4, 10, 10):
        overlap_sweep, overlap_exact = get_overlaps(mult, 8)
        print(np.round(overlap_sweep, 3))
        print(np.round(overlap_exact, 3))
        print(trace_dist(overlap_exact, overlap_sweep))


def probe_js():

    style()
    edm = ExperimentDataManager(
        "probe_sweep_time_vs_ed", start_new_run=False, project="nlce", dry_run=False
    )

    for n_qubits in (3, 4, 5, 6, 7, 8):
        edm.new_run()
        fig, jobj = probe_dispersion_diff_scan_j(n_qubits)
        edm.save_dict(jobj)
        edm.save_figure(fig)


def probe_ks():
    style()
    edm = ExperimentDataManager(
        "probe_sweep_time_vs_ed_kxaxis",
        start_new_run=False,
        project="nlce",
        dry_run=False,
    )

    for n_qubits in (3, 4, 5, 6, 7, 8):
        edm.new_run()
        fig, jobj = probe_dispersion_diff_scan_k(n_qubits)
        edm.save_dict(jobj)
        edm.save_figure(fig)


if __name__ == "__main__":

    probe_ks()
    # probe_ks()
