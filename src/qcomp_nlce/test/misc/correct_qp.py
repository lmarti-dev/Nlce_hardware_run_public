from src.qcomp_nlce.utils import n_qp_indices, to_bitstring
import itertools


n_qubits = 7
n_qp = 2
indices = n_qp_indices(n_qubits, n_qp, True)

print(len(indices), len(list(itertools.combinations(range(n_qubits), n_qp))))

for iii in indices:
    print(f"{iii: <5}{to_bitstring(iii, n_qubits, True)}")
