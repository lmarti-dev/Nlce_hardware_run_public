from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter
import numpy as np


def get_sweep_function(kind: str):
    if kind == "sin":

        def sweep_function(t):
            return np.sin(np.pi / 2 * np.sin(np.pi * t / 2) ** 2) ** 2

    elif kind == "simple":

        def sweep_function(t):
            return t

    else:
        raise NotImplementedError(f"Unknown kind: {kind}")

    return sweep_function


def get_tfim_sweep_circuit(
    ham_j: SparsePauliOp,
    ham_h: SparsePauliOp,
    total_time: float,
    n_steps: int,
    sweep_function: str = "simple",
    synthesis: str = "lie",
) -> QuantumCircuit:

    sf = get_sweep_function(sweep_function)
    if synthesis == "lie":
        synth_fn = LieTrotter()
    elif synthesis == "suzuki":
        synth_fn = SuzukiTrotter()
    circuit = QuantumCircuit((ham_j + ham_h).num_qubits)
    for n in range(n_steps):
        circuit.append(
            PauliEvolutionGate(
                ham_h + ham_j * sf(n / (n_steps - 1)),
                total_time / n_steps,
                synthesis=synth_fn,
            ),
            range((ham_j + ham_h).num_qubits),
        )
    return circuit
