import numpy as np


alpha = 1
n_qubits = 10

psi = np.exp(-alpha * np.arange(1, 2**n_qubits + 1))
prob = psi / np.sum(psi)
psi /= np.linalg.norm(psi)


for Mpow in np.linspace(1, 7, 10):
    M = int(10**Mpow)
    bs = np.random.choice(int(2**n_qubits), M, True, prob)
    calo = np.log(M) / alpha
    unik_bs = len(list(set(bs)))
    print(f"M: {M:.3e}, ub: {unik_bs:n}, approx: {calo:.4f}")
