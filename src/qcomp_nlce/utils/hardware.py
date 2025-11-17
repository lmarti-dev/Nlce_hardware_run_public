from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, RuntimeJob, SamplerV2

try:
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
    from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
except ModuleNotFoundError:
    pass


import io

from qiskit_ibm_runtime.fake_provider import (
    FakeLagosV2,
    FakeWashingtonV2,
    FakeMelbourneV2,
    FakeKolkataV2,
)
from typing import Literal, Union

from qiskit.primitives.base.sampler_result import SamplerResult
from qiskit.circuit.parametertable import ParameterView
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator
from qiskit_aqt_provider.aqt_resource import AQTResource
from mqp.qiskit_provider import MQPProvider, MQPBackend

from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

from qcomp_nlce.utils.circuit import (
    isa_circuit_param_dict,
    parameters_dict,
    parameters_exist,
)
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend

BACKEND_ALIAS = str | FakeBackendV2 | IBMBackend | MQPBackend


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
    n_qubits: int, which: Literal["IBM", "AQT", "IQM"]
) -> Union[IBMBackend, MQPBackend]:
    if which == "IBM":
        return get_ibm_backend(n_qubits)
    elif which == "AQT":
        return get_aqt_backend(n_qubits)
    elif which == "IQM":
        raise NotImplementedError("I'll get to it")
    else:
        raise ValueError(f"Requested invalid backend: {which}")


def get_appropriate_fake_backend(
    n_qubits: int, which: Literal["IBM", "AQT", "IQM"]
) -> Union[FakeBackendV2, AQTResource]:

    if which == "IBM":
        if n_qubits <= 7:
            return FakeLagosV2()
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
    elif which == "IQM":
        if n_qubits <= 5:
            return IQMFakeAdonis()
        elif n_qubits <= 20:
            return IQMFakeGarnet()
        else:
            return IQMFakeAphrodite()
    else:
        raise ValueError(f"Expected a provider got: {which}")


def show_circ_depth(isa_circ: list | QuantumCircuit):
    if isinstance(isa_circ, list):
        if len(isa_circ) == 1:
            print(f"Circuit depth is {isa_circ[0].depth()}")
        else:
            depths = [c.depth() for c in isa_circ]
            mean_depth = np.mean(depths)
            max_depth = np.max(depths)
            print(
                f"Mean circuit depth ({len(isa_circ)}): {mean_depth}, max: {max_depth}"
            )

    elif isinstance(isa_circ, QuantumCircuit):
        print(f"Circuit depth is {isa_circ.depth()}")
    else:
        print(f"circ is neither QuantumCircuit nor list, but: {type(isa_circ)}")


def circuits_to_payload(
    backend,
    circuit: Union[QuantumCircuit, list[QuantumCircuit]],
    parameters: np.ndarray,
    observables: Union[list[SparsePauliOp], SparsePauliOp],
    which: str = "v2",
) -> list[tuple[QuantumCircuit, dict]]:
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)
    show_circ_depth(isa_circuit)
    if which == "v1":
        payload = [[], [], []]
        # assume that for each circuit you want to measure each parameters
        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]
        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        for c in isa_circuit:
            for o in observables:
                payload[0].append(c)
                isa_o = o.apply_layout(c.layout)
                payload[1].append(isa_o)
                if parameters_exist(parameters):
                    payload[2].append(parameters)
        return payload

    elif which == "v2":
        payload = []

        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]

        for idx in range(len(isa_circuit)):
            toadd = []
            if observables is not None:
                toadd.append(
                    [o.apply_layout(isa_circuit[idx].layout) for o in observables]
                )
            if parameters_exist(parameters):
                isa_params = isa_circuit_param_dict(
                    isa_circuit[idx], circuit[idx], parameters
                )
                toadd.append(isa_params)
            payload.append((isa_circuit[idx], *toadd))

    return payload


def backend_sample(
    backend,
    circuit,
    parameters: np.ndarray,
    n_shots: int,
) -> RuntimeJob:
    if isinstance(backend, (IBMBackend, FakeBackendV2)):
        sampler = SamplerV2(mode=backend)

        payload = circuits_to_payload(backend, circuit, parameters, None)
        job = sampler.run(
            payload,
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
    backend: BACKEND_ALIAS,
    observables: Union[SparsePauliOp, list[SparsePauliOp]],
    circuit: Union[QuantumCircuit | list[QuantumCircuit]],
    parameters: np.ndarray,
    n_shots: int,
) -> RuntimeJob:

    if isinstance(backend, AQTResource):
        estimator = AQTEstimator(backend)
        (circuit, observables, parameters) = circuits_to_payload(
            backend, circuit, parameters, observables, which="v1"
        )
        # lose yourself in the moment, the music
        # the moment you own it you never let it go (oh)
        # you only get 2000 shots
        kwargs = {
            "circuits": circuit,
            "observables": observables,
            "shots": min(n_shots, 2000),
        }
        if parameters != []:
            kwargs.update({"parameters": parameters})

        job = estimator.run(**kwargs)
    else:
        estimator = EstimatorV2(backend, options={"resilience_level": 2})
        payload = circuits_to_payload(backend, circuit, parameters, observables)
        job = estimator.run(
            payload,
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
