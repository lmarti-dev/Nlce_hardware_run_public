from qcomp_nlce.utils.folders import res_sim_dir
from qcomp_nlce.utils.circuit import apply_parameters, circ_stats
from qcomp_nlce.utils.sweep import sweep_time_from_steps, get_tfim_sweep_circuit
from qcomp_nlce.utils.save import load_optimized_vqe
from qcomp_nlce.utils.hardware import (
    compile_circuit_for_backend,
    get_appropriate_fake_backend,
)
from qcomp_nlce.utils.hamiltonian import TFIM
from pathlib import Path
from qiskit import QuantumCircuit

from qiskit.transpiler import PassManager
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.synthesis import (
    AILinearFunctionSynthesis,
    AIPauliNetworkSynthesis,
)
from qiskit_ibm_transpiler.ai.collection import (
    CollectLinearFunctions,
    CollectPauliNetworks,
)


def print_stats(circuit: QuantumCircuit, name: str):
    depth, n_gates, nl_gates = circ_stats(circuit)
    print(f"{name}: depth {depth}, total gates: {n_gates}, nonlocal gates: {nl_gates}")


def test_compilation_schemes(circuit: QuantumCircuit):

    backend = get_appropriate_fake_backend(circuit.num_qubits, "IQM")
    no_ai = compile_circuit_for_backend(circuit, backend, ai=False)
    print_stats(no_ai, "Basic, no AI")
    mit_ai = compile_circuit_for_backend(circuit, backend, ai=True)
    print_stats(mit_ai, "with Heuristic + AI")

    ai_pm = PassManager(
        [
            AIRouting(
                backend=backend,
                optimization_level=3,
                layout_mode="optimize",
            ),
            # Collect and synthesize linear functions
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend=backend, local_mode=True),
            # Collect and synthesize Pauli networks
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(backend=backend, local_mode=True),
        ]
    )
    nurai = ai_pm.run(circuit)
    print_stats(nurai, "with only AI")


def vqe_compression():
    print("# VQE")
    gs_dirname_l = Path(res_sim_dir(), f"gs_vqe_100j/vqe/5q_3l")
    gs_dirname_s = Path(res_sim_dir(), f"gs_vqe_100j/vqe/4q_2l")

    _, elem_circuit_l, elem_weights_l, _ = load_optimized_vqe(gs_dirname_l)
    elem_circuit_l = apply_parameters(elem_circuit_l, elem_weights_l[-1])

    _, elem_circuit_s, elem_weights_s, _ = load_optimized_vqe(gs_dirname_s)
    elem_circuit_s = apply_parameters(elem_circuit_s, elem_weights_s[-1])

    print("## large")
    test_compilation_schemes(elem_circuit_l)
    print("## small")
    test_compilation_schemes(elem_circuit_s)


def sweep_compression():
    print("# Sweep")
    J = 1
    n_steps = 10
    n_qubits = 5
    ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )

    print("## f{n_qubits} qubits")
    test_compilation_schemes(circuit)


if __name__ == "__main__":
    vqe_compression()
    # sweep_compression()
