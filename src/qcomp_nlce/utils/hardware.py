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
from typing import Literal, Union

from qiskit.primitives.base.sampler_result import SamplerResult
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler
from qiskit_aqt_provider.aqt_resource import AQTResource
from mqp.qiskit_provider import MQPProvider, MQPBackend

from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

from qcomp_nlce.utils.circuit import isa_circuit_param_dict, parameters_dict
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend


def get_aqt_backend(n_qubits: int) -> MQPBackend:
    if n_qubits > 12:
        raise ValueError(f"Maximum number of qubits is 12, requested: {n_qubits}")
    token = get_token("LRZ")
    provider = MQPProvider(token)
    backend = provider.get_backend("AQT20")
    return backend


def sampler_result_to_json(sampler: SamplerResult):
    d = {}

    d["quasi_dists"] = [
        dict(sampler.quasi_dists[m]) for m in range(len(sampler.quasi_dists))
    ]
    d["metadata"] = sampler.metadata

    return d


def get_ibm_backend(n_qubits: int) -> IBMBackend:
    token = get_token("IBM")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    real_backend = service.least_busy(
        simulator=False, operational=True, min_num_qubits=n_qubits
    )
    return real_backend


def get_backend(
    n_qubits: int, which: Literal["IBM", "AQT"]
) -> Union[IBMBackend, MQPBackend]:
    if which == "IBM":
        return get_ibm_backend(n_qubits)
    elif which == "AQT":
        return get_aqt_backend(n_qubits)
    else:
        raise ValueError(f"Requested invalid backend: {which}")


def get_appropriate_fake_backend(
    n_qubits: int, which: Literal["IBM", "AQT"]
) -> Union[FakeBackendV2, AQTResource]:

    if which == "IBM":
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
    elif which == "AQT":
        if n_qubits > 12:
            raise ValueError(f"Maximum number of qubits is 12, requested: {n_qubits}")
        return AQTProvider("ANY_TOKEN_WORKS_FOR_OFFLINE_SIM").get_backend(
            "offline_simulator_noise"
        )
    else:
        raise ValueError(f"Expected a provider got: {which}")


def backend_sample(
    backend,
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
) -> RuntimeJob:
    if isinstance(backend, (IBMBackend, FakeBackendV2)):
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

    elif isinstance(backend, AQTResource):
        circuit = circuit.assign_parameters(parameters_dict(circuit, parameters))
        sampler = AQTSampler(backend)
        job = sampler.run([circuit], parameter_values=None, shots=n_shots)
    elif isinstance(backend, MQPBackend):
        circuit = circuit.assign_parameters(parameters_dict(circuit, parameters))
        job = backend.run(circuit, shots=n_shots, qasm3=True, queued=True)
    else:
        raise ValueError(f"Unexpected backend: type {type(backend)}")
    return job


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


def get_token(which: Literal["IBM", "LRZ"]):
    home = Path(__file__).parent
    if which == "IBM":
        token_fpath = Path(home, "../files/ibm_token.dat")
    elif which == "LRZ":
        token_fpath = Path(home, "../files/lrz_token.dat")
    else:
        raise ValueError(f"Unallowed value: {which}")
    token = io.open(
        token_fpath,
        "r",
        encoding="utf8",
    ).read()

    return token


def encode_sampler_result(job):
    res = job.result()
    if isinstance(res, SamplerResult):
        return sampler_result_to_json(res)
    else:
        return res
