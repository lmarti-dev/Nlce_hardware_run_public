import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
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
        e = fun(xk=xk)
        print(e, end="\r")
        energy_values.append(e)
        xks.append(xk)

    def fun(xk: np.ndarray):
        result = estimate_observables(
            circuit=circuit, observables=[hamiltonian], param_sweep=xk
        )
        return result.data.evs[0]

    def jac(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for ind in range(len(x)):
            p = x.copy()
            p[ind] += np.pi / 2
            m = x.copy()
            m[ind] -= np.pi / 2
            res = estimate_observables(
                circuit=circuit, observables=[hamiltonian], param_sweep=[p, m]
            )
            out[ind] = 0.5 * (res.data.evs[1] - res.data.evs[0])
        return out

    minimize(
        fun=fun,
        x0=np.zeros((len(circuit.parameters))),
        method="trust-constr",
        callback=callback,
        bounds=circuit.num_parameters * [(0, np.pi)],
    )
    return energy_values, xks


def get_circuit_energy(hamiltonian: SparsePauliOp, circuit: QuantumCircuit):
    return Statevector(circuit).expectation_value(hamiltonian)
