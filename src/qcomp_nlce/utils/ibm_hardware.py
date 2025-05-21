from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
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


def isa_circuit_param_dict(
    new_circuit: QuantumCircuit, old_circuit: QuantumCircuit, parameters: np.ndarray
) -> dict:
    param_dict = {k: v for k, v in zip(old_circuit.parameters, parameters)}
    return {k: param_dict[k] for k in new_circuit.parameters}


def backend_sample(
    backend,
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
) -> RuntimeJob:
    sampler = SamplerV2(mode=backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)
    if parameters is None:
        payload = isa_circuit
    else:
        payload = (
            isa_circuit,
            isa_circuit_param_dict(isa_circuit, circuit, parameters),
        )
    job = sampler.run(
        [payload],
        shots=int(n_shots),
    )
    return job


def get_unique_bitstrings(counts: dict) -> list[str]:
    return np.unique(list(counts.keys()))


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
                isa_circuit_param_dict(isa_circuit, circuit, parameters),
            )
        ],
        precision=1 / np.sqrt(n_shots),
    )
    return job


def get_ibm_token():
    home = Path(__file__).parent
    token_fpath = Path(home, "../files/ibm_token.dat")
    token = io.open(
        token_fpath,
        "r",
        encoding="utf8",
    ).read()

    return token
