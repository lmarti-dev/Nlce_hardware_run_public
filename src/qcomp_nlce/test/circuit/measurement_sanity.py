from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.random import random_circuit
from qcomp_nlce.utils import circuit_to_matrix
import numpy as np


def mat_val(circ_mat: np.ndarray, op: SparsePauliOp):
    opmat = op.to_matrix()
    l1 = np.zeros(len(circ_mat))
    l1[0] = 1
    v = circ_mat @ l1
    return np.real(v.conj().dot(opmat @ v))


def trace_out_shots(counts: dict, inds: list[int]):
    # assuming z-basis sampled
    val = 0
    tot = 0
    for k in counts:
        tot += counts[k]
        subs = sum([int(k[jj]) for jj in inds])
        val += counts[k] * (-1) ** subs
    return val / tot


n_qubits = 10
n_shots = int(1e5)

odd = {}
even = {}
zs = {}

for jj in range(n_qubits - 1):

    ll = ["I" for _ in range(n_qubits)]
    ll[jj] = "X"
    ll[jj + 1] = "X"
    op = SparsePauliOp("".join(ll))
    if jj % 2 == 0:
        even[(jj, jj + 1)] = op
    else:
        odd[(jj, jj + 1)] = op

for jj in range(n_qubits):
    ll = ["I" for _ in range(n_qubits)]
    ll[jj] = "Z"
    op = SparsePauliOp("".join(ll))
    zs[(jj,)] = op


circ = random_circuit(n_qubits, 10, 2)
circ_mat = circuit_to_matrix(circ)

circ_x: QuantumCircuit = circ.copy()
circ_x.h(range(n_qubits))
circ_x.measure_all()

sampler = StatevectorSampler()


job = sampler.run([circ_x], shots=n_shots)
x_samples = job.result()[0].join_data().get_counts()


for d in (odd, even):
    for y in d:
        true_val = mat_val(circ_mat, d[y])
        circ_val = trace_out_shots(x_samples, y)

        print(
            f"{str(d[y][0].paulis[0])} diff: {np.abs(true_val-circ_val):.5f} true: {true_val:.5f}, meas: {circ_val:.5f}"
        )
