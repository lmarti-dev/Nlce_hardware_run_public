import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
from scipy.optimize import minimize


def add_meas_layer(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    meas_register = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(meas_register)
    for q, c in zip(circuit.qubits, meas_register):
        circuit.measure(q, c)
    return circuit


def estimate_observables(
    circuit: QuantumCircuit, observables: list[SparsePauliOp], param_sweep: np.ndarray
):
    estimator = StatevectorEstimator()
    job_observables = []
    for ind, o in enumerate(observables):
        if o.num_qubits < circuit.num_qubits:
            job_observables.append(
                SparsePauliOp("I" * (circuit.num_qubits - o.num_qubits)).expand(o)
            )
        else:
            job_observables.append(o)
    pub = (circuit, job_observables, param_sweep)
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
        method="COBYLA",
        callback=callback,
        bounds=circuit.num_parameters * [(0, np.pi)],
    )
    return energy_values, xks


def get_circuit_energy(
    hamiltonian: SparsePauliOp, circuit: QuantumCircuit, use_estimator: bool = False
):
    if use_estimator:
        return estimate_observables(circuit, [hamiltonian], None).data.evs[0]
    else:
        return Statevector(circuit).expectation_value(
            hamiltonian,
            range(0, hamiltonian.num_qubits),
        )


def statevector_sample_unique(
    circuit: QuantumCircuit,
    n_shots: int,
    qubits: list = None,
    use_sampler: bool = False,
) -> list:
    out_circ = circuit.remove_final_measurements(inplace=False)
    out_circ.measure_all(add_bits=True)
    if qubits is None:
        qubits = list(range(circuit.num_qubits))
    if use_sampler:
        sampler = StatevectorSampler(seed=0)
        job = sampler.run([out_circ], shots=n_shots)
        counts = job.result()[0].data.meas.get_counts()
    else:
        sv = Statevector(circuit)
        if len(qubits) < circuit.num_qubits:
            sv = partial_trace(
                sv, [q for q in range(circuit.num_qubits) if q not in qubits]
            )
        counts = sv.sample_counts(n_shots)
    res = [str(k) for k in counts.keys()]
    return res
