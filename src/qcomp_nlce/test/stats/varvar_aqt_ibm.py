from src.qcomp_nlce.utils import (
    get_appropriate_fake_backend,
    backend_estimate,
    backend_sample,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np


def get_zstd(data: np.ndarray):
    return np.sqrt(1 - (np.abs(data[0]) ** 2 - np.abs(data[1]) ** 2) ** 2)


c = QuantumCircuit(1)
c.rx(0.342, 0)
n_shots = 1000
n_qubits = c.num_qubits
sv = Statevector(c)
ham = SparsePauliOp("Z")

print(
    f"sv: {sv.data[0]:.3f},{sv.data[1]:.3f}, error-on-the-mean: {get_zstd(sv.data) / np.sqrt(n_shots):.5f}"
)

for whichbe in ("IBM", "AQT"):

    backend = get_appropriate_fake_backend(n_qubits, whichbe)

    estim_job = backend_estimate(backend, ham, c, None, n_shots)

    estim_res = estim_job.result()

    if whichbe == "AQT":
        mean = estim_res.values[0]
        std = (estim_res.metadata[0]["variance"] ** (1 / 2)) / np.sqrt(n_shots)
    else:
        mean = estim_res[0].data.evs[0]
        std = estim_res[0].data.stds[0]
    print(f"{whichbe} Estim: {mean:.5f}, std: {std:.5f}")
