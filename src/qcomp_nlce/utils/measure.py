from qiskit.primitives import StatevectorSampler, PrimitiveJob, StatevectorEstimator
from qcomp_nlce.utils.hardware import backend_sample, backend_estimate, BACKEND_ALIAS
from qcomp_nlce.utils.circuit import parameters_exist
from qiskit_ibm_runtime import IBMBackend, RuntimeJob
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit import ClassicalRegister, QuantumCircuit
from typing import Union
from mqp.qiskit_provider.job import MQPJob
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit_aer import AerJob
from iqm.qiskit_iqm.iqm_job import IQMJob


def estimate_circuit(
    circuit: QuantumCircuit,
    observables: Union[SparsePauliOp, list[SparsePauliOp]],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
):
    if backend == "statevector":
        estimator = StatevectorEstimator()

        if isinstance(circuit, QuantumCircuit):
            circuit = [circuit]
        ctr = [[c, observables] for c in circuit]
        if parameters_exist(parameters):
            for idx in range(len(ctr)):
                ctr[idx].append(parameters)
        job = estimator.run(ctr, precision=1 / np.sqrt(n_shots))
    else:
        job = backend_estimate(
            backend=backend,
            observables=observables,
            circuit=circuit,
            parameters=parameters,
            n_shots=n_shots,
        )
    return job


def sample_circuit(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
) -> PrimitiveJob:
    if backend == "statevector":
        # statevector always sample the same outcomes if seed is set
        # I guess this is for reproducibility
        # Anyway, don't set seed unless you want to pull your hair out
        # https://github.com/Qiskit/qiskit/issues/13047
        sampler = StatevectorSampler()
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


def get_counts_from_res(res: Union[RuntimeJob, PrimitiveJob, MQPJob]) -> dict:
    return res.join_data().get_counts()


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
            counts = res[ind].join_data().get_counts()
    elif isinstance(job, (MQPJob, AerJob, IQMJob)):
        counts = job.result().get_counts(ind)
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
