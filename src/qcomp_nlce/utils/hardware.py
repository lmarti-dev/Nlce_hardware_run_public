import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, RuntimeJob


def backend_estimate(
    backend,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
) -> RuntimeJob:
    estimator = EstimatorV2(backend, options={"resilience_level": 2})

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)
    layout = isa_circuit.layout
    isa_observables = hamiltonian.apply_layout(layout)
    job = estimator.run(
        [(isa_circuit, isa_observables, parameters)], precision=1 / np.sqrt(n_shots)
    )
    return job
