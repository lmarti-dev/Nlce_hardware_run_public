import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
from scipy.optimize import minimize
from itertools import combinations
from collections.abc import Callable
from qcomp_nlce.utils.circuit import apply_parameters


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


def optimize_circuit(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    cost: Callable[[np.ndarray], float] = None,
) -> list:
    energy_values = []
    xks = []

    def callback(xk: np.ndarray, *args):
        e = cost(xk=xk)
        print(e, end="\r")
        energy_values.append(e)
        xks.append(xk)

    if cost is None:

        def cost(xk: np.ndarray):
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
        fun=cost,
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


def statevector_sample(circuit: QuantumCircuit, n_shots: int):
    sampler = StatevectorSampler(seed=0)
    job = sampler.run([circuit], shots=n_shots)
    counts = job.result()[0].data.meas.get_counts()
    return counts


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


def hamming_weight_states(n_qubits: int, hamming_weight: int) -> list[str]:
    comps = combinations(range(n_qubits), hamming_weight)
    states = []
    for comp in comps:
        state = ["0"] * n_qubits
        for c in comp:
            state[c] = "1"
        states.append(state)
    return states


def estimate_observable_1qp_initial_state(
    circuit: QuantumCircuit, parameters: np.ndarray, observable: SparsePauliOp
) -> np.ndarray:
    n_qubits = circuit.num_qubits

    out = np.zeros((n_qubits, n_qubits))

    for ind_y in range(n_qubits):
        for ind_x in range(n_qubits):
            # initial state
            circ_in = QuantumCircuit(circuit.num_qubits)
            circ_in.x(ind_y)
            circ_in = circ_in.compose(circuit)
            # U1.compose(U2) gives U2 U1 |psi>
            circ_out = QuantumCircuit(circuit.num_qubits)
            # x is left bra and y right ket
            circ_out.x(ind_x)
            circ_out = circ_in.compose(circ_out, range(circuit.num_qubits))
            sv = Statevector(apply_parameters(circuit, parameters))
            out[ind_x, ind_y] = np.real(sv.expectation_value(observable))
    return out
 # line 147, circuit needs to be replaced with circ_out