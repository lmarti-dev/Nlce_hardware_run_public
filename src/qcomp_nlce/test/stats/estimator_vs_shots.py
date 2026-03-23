from src.qcomp_nlce.utils import (
    get_appropriate_fake_backend,
    backend_estimate,
    backend_sample,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np


def estimc(circuit, n_shots):

    estim_job = backend_estimate(backend, SparsePauliOp("Z"), circuit, None, n_shots)

    estim_res = estim_job.result()

    print(
        f"Estim: {estim_res.values[0]:.5f}, std: {estim_res.metadata[0]['variance']**(1/2):.5f}"
    )


c = QuantumCircuit(1)
c.rx(0.342, 0)
n_qubits = c.num_qubits
sv = Statevector(c)

print(f"sv: {sv.data[0]:.3f},{sv.data[1]:.3f}")


backend = get_appropriate_fake_backend(n_qubits, "AQT")


for n_shots in range(10, 10000, 100):
    estimc(c, n_shots)
