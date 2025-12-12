from qcomp_nlce.utils import get_backend, backend_estimate
from qiskit import QuantumCircuit
from data_manager import ExperimentDataManager
from qiskit.visualization import plot_distribution
import os, json
from pathlib import Path
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp


def w_circ(n_qubits: int):
    c = QuantumCircuit(n_qubits)
    c.h(0)
    for n in range(n_qubits - 1):
        c.cx(n, n + 1)
    c.measure_all()
    return c


def hardware_run():
    n_qubits = 5
    circ = w_circ(n_qubits)
    edm = ExperimentDataManager("w_test_estim_v2", project="nlce", dry_run=False)
    ops = [SparsePauliOp("X" * n_qubits), SparsePauliOp("Z" * n_qubits)]

    for provider_name, n_shots in (("QEXA", int(2000)),):
        backend = get_backend(n_qubits, provider_name)
        job = backend_estimate(backend, ops, circ, None, n_shots)
        res = job.result()
        for ind, op in enumerate(ops):
            mean = res[0].data.evs[ind]
            std = res[0].data.stds[ind]
            edm.save_dict({"ops": str(op.paulis[0]), "mean": mean, "std": std})
            print(f"{str(op.paulis[0])} mean: {mean:.5f}, std: {std:.5f}")
        edm.var_dump(provider_name=provider_name, n_shots=n_shots)


if __name__ == "__main__":

    hardware_run()
