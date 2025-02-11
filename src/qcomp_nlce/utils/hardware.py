from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, RuntimeJob, SamplerV2
import io

from qiskit_ibm_runtime.fake_provider import (
    FakeJakartaV2,
    FakeManilaV2,
    FakeWashingtonV2,
    FakeMelbourneV2,
    FakeKolkataV2,
)
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2


def get_appropriate_fake_backend(n_qubits: int) -> FakeBackendV2:

    if n_qubits <= 5:
        return FakeManilaV2()
    elif n_qubits <= 7:
        return FakeJakartaV2()
    elif n_qubits <= 14:
        return FakeMelbourneV2()
    elif n_qubits <= 27:
        return FakeKolkataV2()
    else:
        return FakeWashingtonV2()


def add_meas_layer(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    meas_register = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(meas_register)
    for q, c in zip(circuit.qubits, meas_register):
        circuit.measure(q, c)
    return circuit


def backend_sample(
    backend,
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
) -> RuntimeJob:
    sampler = SamplerV2(mode=backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    job = sampler.run(
        [(isa_circuit, {k: v for k, v in zip(circuit.parameters, parameters)})],
        shots=int(n_shots),
    )
    return job


def get_counts_from_job(job: RuntimeJob) -> np.array:
    return job.result()[0].data.c0.array.T[0]


def get_unique_bitstrings(shots: np.ndarray) -> list[str]:
    return np.unique(shots)


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
        [
            (
                isa_circuit,
                isa_observables,
                {k: v for k, v in zip(circuit.parameters, parameters)},
            )
        ],
        precision=1 / np.sqrt(n_shots),
    )
    return job


def get_token():
    home = Path(__file__).parent
    token_fpath = Path(home, "../files/token.dat")
    token = io.open(
        token_fpath,
        "r",
        encoding="utf8",
    ).read()

    return token
