from qcomp_nlce.utils import statevector_sample_unique
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

circuit = QuantumCircuit(3)
circuit.append(
    PauliEvolutionGate(
        SparsePauliOp("XX"),
        2,
    ),
    range(2),
)
circuit2 = QuantumCircuit(4)
circuit2.compose(circuit, range(circuit.num_qubits), inplace=True)


bitstrings1 = statevector_sample_unique(circuit, 1000)
print(bitstrings1)
bitstrings2 = statevector_sample_unique(circuit2, 1000)
print(bitstrings2)
