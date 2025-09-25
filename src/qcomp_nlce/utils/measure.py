from qiskit.primitives import StatevectorSampler, PrimitiveJob
from qcomp_nlce.utils.hardware import backend_sample
from qiskit_ibm_runtime import IBMBackend, RuntimeJob
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit import ClassicalRegister, QuantumCircuit
from typing import Union
from mqp.qiskit_provider.job import MQPJob

import numpy as np


def sample_circuit(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    backend: str | FakeBackendV2 | IBMBackend = "statevector",
    n_shots: int = 1000,
) -> PrimitiveJob:
    if backend == "statevector":
        sampler = StatevectorSampler(seed=0)
        if isinstance(circuit, QuantumCircuit):
            ctr = [circuit]
        else:
            ctr = circuit
        job = sampler.run(ctr, shots=int(n_shots))
    else:
        job = backend_sample(
            backend=backend, circuit=circuit, parameters=parameters, n_shots=n_shots
        )
    return job


def sample_counts_circuit(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    backend: str | FakeBackendV2 | IBMBackend = "statevector",
    n_shots: int = 1000,
):
    job = sample_circuit(circuit, parameters, backend, n_shots)
    counts = get_counts_from_job(job)
    return counts


def get_counts_from_res(res) -> dict:
    return res.get_counts()


def get_counts_from_job(
    job: Union[RuntimeJob, PrimitiveJob, MQPJob], ind: int = 0
) -> dict:
    if isinstance(job, RuntimeJob):
        key = list(job.result()[ind].data.keys())[0]
        counts = job.result()[ind].data[key].get_counts()
    elif isinstance(job, PrimitiveJob):
        res = job.result()
        try:
            qd = res.quasi_dists[ind]
            n_shots = res.metadata[ind]["shots"]
            counts = {k: int(qd[k] * n_shots) for k in qd.keys()}
        except AttributeError:
            counts = res[ind].data.meas.get_counts()
    elif isinstance(job, MQPJob):
        counts = job.result().get_counts()
    else:
        raise ValueError(f"Unexpected job type: type {type(job)}")
    return counts


def add_meas_layer(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    meas_register = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(meas_register)
    for q, c in zip(circuit.qubits, meas_register):
        circuit.measure(q, c)
    return circuit


def get_unique_bitstrings(counts: dict) -> list[str]:
    return np.unique(list(counts.keys()))
