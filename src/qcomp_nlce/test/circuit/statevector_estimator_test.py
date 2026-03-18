from qiskit.primitives import StatevectorEstimator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import numpy as np


qc = QuantumCircuit(1)
qc.h(0)

estimator = StatevectorEstimator()

for spow in (0,) * 10:
    n_shots = 10**spow
    job = estimator.run(
        [
            (qc, SparsePauliOp("Z")),
        ],
        precision=1 / np.sqrt(n_shots),
    )
    res = job.result()

    v = res[0].data.evs
    s = res[0].data.stds
    comps = np.sqrt(1 - v**2)
    print(f"{n_shots}, {v}, {s}, {comps}")
