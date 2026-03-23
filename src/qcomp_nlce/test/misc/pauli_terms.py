from src.qcomp_nlce.utils import get_pauli_indices, TFIM


ham = TFIM(n_qubits=4, h=1, J=0.7)


for term in ham:
    p = term.paulis[0]
    idx = get_pauli_indices(p)

    print(p, idx)

    print("".join([f"{x}{y}" for x, y in zip(p[::-1][idx], idx)]))
