from src.qcomp_nlce.utils import (
    get_appropriate_fake_backend,
    backend_estimate,
    backend_sample,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np


def sampleres_vals(sample_res):
    qd = sample_res.quasi_dists[0]
    mean = qd[0] - qd[1]
    std = np.sqrt(qd[0] * ((1 - mean) ** 2 + qd[1] * (-1 - mean) ** 2))
    return mean, std


c = QuantumCircuit(1)
c.rx(0.342, 0)
n_shots = 1000
n_qubits = c.num_qubits
sv = Statevector(c)

print(f"sv: {sv.data[0]:.3f},{sv.data[1]:.3f}")

c2: QuantumCircuit = c.copy()
c2.measure_all()

backend = get_appropriate_fake_backend(n_qubits, "AQT")

estim_job = backend_estimate(backend, SparsePauliOp("Z"), c, None, n_shots)
sample_job = backend_sample(backend, c2, None, n_shots)


estim_res = estim_job.result()
sample_res = sample_job.result()


smean, sstd = sampleres_vals(sample_res)

print(
    f"Estim: {estim_res.values[0]:.5f}, std: {estim_res.metadata[0]['variance']**(1/2):.5f}"
)
print(f"Estim: {smean:.5f}, std: {sstd:.5f}")

...
