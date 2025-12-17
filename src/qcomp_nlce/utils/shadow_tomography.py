from povm_toolbox.library import ClassicalShadows
from povm_toolbox.sampler import POVMSampler
from povm_toolbox.post_processor import POVMPostProcessor
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qcomp_nlce.utils.folders import res_sim_dir
from pathlib import Path
from qcomp_nlce.utils.save import load_optimized_vqe
from qcomp_nlce.utils.circuit import apply_parameters, circuit_matrix_elements
from qcomp_nlce.utils.hardware import (
    BACKEND_ALIAS,
    get_backend_sampler,
    get_appropriate_fake_backend,
    compile_circuit_for_backend,
)
from qiskit_ibm_transpiler import generate_ai_pass_manager
from qcomp_nlce.utils.hamiltonian import projector_ham, trace_dist
from qcomp_nlce.utils.statevector import hamming_weight_states
import numpy as np


def single_pauli(j: int, n_qubits: int, which: str):
    if j == 0:
        return SparsePauliOp(which + "I" * (n_qubits - 1))
    elif j == (n_qubits - 1):

        return SparsePauliOp("I" * (n_qubits - 1) + which)
    else:
        return SparsePauliOp("I" * (j) + which + "I" * (n_qubits - j - 1))


def split_obs_real_imag(op: SparsePauliOp) -> tuple[SparsePauliOp, SparsePauliOp]:
    op1 = 0
    op2 = 0
    if np.all([term.coeffs[0] == term.coeffs[0].imag for term in op]):
        return None, op
    elif np.all([term.coeffs[0] == term.coeffs[0].real for term in op]):
        return op, None
    for term in op:
        c = term.coeffs[0]
        p = str(term.paulis[0])
        op1 += SparsePauliOp.from_list(
            [
                (p, c.real),
            ]
        )
        op2 += SparsePauliOp.from_list(
            [
                (p, c.imag),
            ]
        )
    return op1.simplify(), op2.simplify()


def jk_obs(j: int, k: int, n_qubits: int) -> SparsePauliOp:
    xj = single_pauli(j, n_qubits, "X")
    xk = single_pauli(k, n_qubits, "X")
    proj = projector_ham(n_qubits)
    return (xj @ proj @ xk).simplify()


def sv_classical_shadows(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    n_shots: int = 16,
):
    povm = ClassicalShadows(observable.num_qubits)
    sampler = StatevectorSampler()
    povm_sampler = POVMSampler(sampler)
    job = povm_sampler.run([circuit], povm=povm, shots=n_shots)
    result = job.result()
    post_processor = POVMPostProcessor(result[0])
    val, std = post_processor.get_expectation_value(observable)
    return val, std


def backend_classical_shadows(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: BACKEND_ALIAS,
    n_shots: int = 16,
):
    povm = ClassicalShadows(observable.num_qubits)

    # need transpile AND pass manager for IQM
    pm = generate_ai_pass_manager(
        ai_optimization_level=3,
        optimization_level=3,
        ai_layout_mode="optimize",
        backend=backend,
    )
    sampler = get_backend_sampler(backend)
    povm_sampler = POVMSampler(sampler)

    circuit = compile_circuit_for_backend(
        circuit, backend, {"optimization_level": 3}, False
    )

    job = povm_sampler.run([circuit], povm=povm, shots=n_shots, pass_manager=pm)
    result = job.result()
    post_processor = POVMPostProcessor(result[0])
    val, std = post_processor.get_expectation_value(observable)
    return val, std


def backend_classical_shadows_matrix_elements(
    circuit: QuantumCircuit,
    backend: BACKEND_ALIAS,
    n_shots: int = 16,
):

    n_qubits = circuit.num_qubits
    povm = ClassicalShadows(n_qubits)
    if backend == "statevector":

        sampler = StatevectorSampler()
        kwargs = {}

    else:
        pm = generate_ai_pass_manager(
            ai_optimization_level=3,
            optimization_level=3,
            ai_layout_mode="optimize",
            backend=backend,
        )
        sampler = get_backend_sampler(backend)
        circuit = compile_circuit_for_backend(
            circuit, backend, {"optimization_level": 3}, False
        )
        kwargs = {"pass_manager": pm}
    povm_sampler = POVMSampler(sampler)
    job = povm_sampler.run([circuit], povm=povm, shots=n_shots, **kwargs)
    result = job.result()
    post_processor = POVMPostProcessor(result[0])
    out = np.zeros((n_qubits, n_qubits), dtype=complex)
    stds = np.zeros((n_qubits, n_qubits), dtype=complex)
    for j in range(n_qubits):
        for k in range(n_qubits):
            observable = jk_obs(j, k, n_qubits)
            shadow_norm, chunk_size = shadow_bound(1e-2, [observable.to_matrix()])
            print(f"Number of samples needed: {shadow_norm:.5f}")
            op_r, op_i = split_obs_real_imag(observable)
            if op_r is None:
                val_r, std_r = 0, 0
            else:
                val_r, std_r = post_processor.get_expectation_value(op_r)
            if op_i is None:
                val_i, std_i = 0, 0
            else:
                val_i, std_i = post_processor.get_expectation_value(op_i)
            out[j, k] = val_r + 1j * val_i
            stds[j, k] = std_r + 1j * std_i

    return out, stds


# from https://pennylane.ai/qml/demos/tutorial_classical_shadows
def shadow_bound(error: float, observables: list[np.ndarray], failure_rate=0.01):
    """
    Calculate the shadow bound for the Pauli measurement scheme.

    Implements Eq. (S13) from https://arxiv.org/pdf/2002.08953.pdf

    Args:
        error (float): The error on the estimator.
        observables (list) : List of matrices corresponding to the observables we intend to
            measure.
        failure_rate (float): Rate of failure for the bound to hold.

    Returns:
        An integer that gives the number of samples required to satisfy the shadow bound and
        the chunk size required attaining the specified failure rate.
    """

    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error**2
    return int(np.ceil(N * K)), int(K)


def compare_vqe_gs():
    gs_dirname_l = Path(res_sim_dir(), f"gs_vqe_100j/vqe/5q_3l")
    n_shots = 32

    hamiltonian, gs_circ, weights, ground_energy = load_optimized_vqe(gs_dirname_l)
    gs_circ = apply_parameters(gs_circ, weights[-1])
    provider_name = "IQM"
    backend = get_appropriate_fake_backend(hamiltonian.num_qubits, provider_name)
    be_val, be_std = backend_classical_shadows(gs_circ, hamiltonian, backend, n_shots)

    sv_val, sv_std = sv_classical_shadows(gs_circ, hamiltonian, n_shots)

    print(f"sv povm: {sv_val:.5f} ({sv_std:.5f}), gs: {ground_energy:.5f}")
    print(f"{provider_name} povm: {be_val:.5f} ({be_std:.5f}), gs: {ground_energy:.5f}")


def compare_elem_gs():
    elem_dirname_l = Path(res_sim_dir(), f"elem_vqe_100j/vqe/cost_offdiag/5q_3l")
    n_shots = 32

    hamiltonian, circ, weights, ground_energy = load_optimized_vqe(elem_dirname_l)
    circ = apply_parameters(circ, weights[-1])
    provider_name = "IQM"
    backend = get_appropriate_fake_backend(hamiltonian.num_qubits, provider_name)
    be_outs, be_stds = backend_classical_shadows_matrix_elements(circ, backend, n_shots)
    sv_outs, sv_stds = backend_classical_shadows_matrix_elements(
        circ, "statevector", n_shots
    )
    comp_states = hamming_weight_states(hamiltonian.num_qubits, 1)
    true_comp_vals = circuit_matrix_elements(circ, comp_states)

    print("sv-true", trace_dist(sv_outs, true_comp_vals))
    print("be-true", trace_dist(be_outs, true_comp_vals))

    ...


if __name__ == "__main__":
    compare_elem_gs()
