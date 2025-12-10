from qiskit import transpile, QuantumCircuit
from qiskit_aqt_provider import AQTProvider

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

backend = AQTProvider("ANY_TOKEN_WORKS_FOR_OFFLINE_SIM").get_backend(
    "offline_simulator_noise"
)
circuit = QuantumCircuit(2)
circuit.append(
    PauliEvolutionGate(
        SparsePauliOp("XX"),
        2,
    ),
    range(2),
)
initial_layout = list(range(circuit[0].num_qubits))
transpile_circuit = transpile(
    circuit,
    backend=backend,
    initial_layout=initial_layout,
    optimization_level=3,
    coupling_map=backend.coupling_map,
)

pm = generate_preset_pass_manager(backend=backend)
isa_circuit = pm.run(circuit)
