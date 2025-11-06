from qcomp_nlce.utils.measure import (
    sample_counts_circuit,
    sample_circuit,
    get_counts_from_job,
)
from qiskit import QuantumCircuit


ind = 0
circ = QuantumCircuit(3)
circ.x(ind)

circ.measure_all()


counts = sample_counts_circuit(circ, [], n_shots=1000)

print(f"The index {ind} has X and should be 1: {counts}")
