from qiskit import QuantumCircuit
from src.qcomp_nlce.utils import sample_counts_circuit
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import time
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2

from qiskit_ibm_runtime import SamplerV2


# results:
# for

# depth = 1000
# n_circs = 8
# n_qubits = 10
# backend = FakeMelbourneV2()
# n_shots = 1000

# Each circuit in turn: 656.9601271152496
# All circuits at once: 114.81504988670349


def sample(backend: FakeBackendV2, circuit, n_shots: int):
    sampler = SamplerV2(mode=backend)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)

    if isinstance(isa_circuit, QuantumCircuit):

        job = sampler.run(
            [isa_circuit],
            shots=int(n_shots),
        )
    else:
        job = sampler.run(
            isa_circuit,
            shots=int(n_shots),
        )
    return job


if __name__ == "__main__":

    depth = 100
    n_circs = 6
    n_qubits = 10
    backend = FakeMelbourneV2()
    n_shots = 1000

    circuits = [random_circuit(n_qubits, depth, measure=True) for _ in range(n_circs)]

    start = time.time()
    jobs = []
    for c in circuits:
        job1 = sample(backend, c, n_shots)
        jobs.append(job1)
    end = time.time()

    print(f"Each circuit in turn: {end-start}")

    start = time.time()
    job2 = sample(backend, circuits, n_shots)
    end = time.time()

    print(f"All circuits at once: {end-start}")
