import matplotlib.pyplot as plt
from qcomp_nlce.utils import TFIM, get_sweep_spectrum, field_pauli_sum
import numpy as np
from fauplotstyle import style
from data_manager import ExperimentDataManager

style()
n_qubits = 4
h = 1
J = 1
ham_j, ham_h = TFIM(n_qubits=n_qubits, h=h, J=J, split_parts=True)

ham_x = field_pauli_sum(n_qubits=n_qubits, pauli="X")

n_steps = 2000
spectrum = get_sweep_spectrum(ham_h + 0.1 * ham_x, ham_j + ham_h + 0.1 * ham_x, n_steps)


fig, ax = plt.subplots()

n_levels = 2**ham_j.num_qubits

for n in range(n_levels):
    ax.plot(np.linspace(0, 1, n_steps), spectrum[n, :])


ax.set_xlabel(r"Normalized sweep parameter")
ax.set_ylabel(r"Energy")

edm = ExperimentDataManager(
    "plot_ising_longfield_model_start_lf", project="nlce", dry_run=True
)

edm.var_dump(j=J, h=h, n_qubits=n_qubits)
edm.save_figure(fig, fig_shape="double-size")

plt.show()
