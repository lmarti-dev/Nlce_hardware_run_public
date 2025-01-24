import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


def estimate_observables(
    circuit: QuantumCircuit, observables: list, param_sweep: np.ndarray
):
    estimator = StatevectorEstimator()
    pub = (circuit, observables, param_sweep)
    job = estimator.run([pub])
    result = job.result()[0]
    return result


def optimize_circuit(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> list:
    energy_values = []
    xks = []

    def callback(xk: np.ndarray, *args):
        energy_values.append(fun(xk=xk))
        xks.append(xk)

    def fun(xk: np.ndarray):
        result = estimate_observables(
            circuit=circuit, observables=[hamiltonian], param_sweep=xk
        )
        return result.data.evs[0]

    minimize(
        fun=fun,
        x0=np.zeros((len(circuit.parameters))),
        method="trust-constr",
        callback=callback,
        bounds=circuit.num_parameters * [(0, np.pi)],
    )
    return energy_values, xks
