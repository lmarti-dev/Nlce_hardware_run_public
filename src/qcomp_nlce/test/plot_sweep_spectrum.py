import matplotlib.pyplot as plt
from qcomp_nlce.utils import get_tfim_ham, get_sweep_spectrum, get_singlet_ham
import numpy as np


n_qubits = 10
h = 1
J = 1
ham_j, ham_h = get_tfim_ham(n_qubits=n_qubits, h=h, J=J, split_parts=True)


n_steps = 20
spectrum = get_sweep_spectrum(ham_h, ham_j + ham_h, n_steps)


fig, ax = plt.subplots()

n_levels = 2**ham_j.num_qubits

for n in range(n_levels):
    ax.plot(np.linspace(0, 1, n_steps), spectrum[n, :])

plt.show()
