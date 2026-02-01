from qcomp_nlce.utils import heff_n_expectation_values_1d
from qcomp_nlce.utils import heff_n_expectation_values_2d

import matplotlib.pyplot as plt
from fauplotstyle import style
from data_manager import ExperimentDataManager


dry_run = True

edm = ExperimentDataManager(
    "expectation_values_scaling", project="nlce", dry_run=dry_run
)


style()
x = range(3, 100)
y = []
z = []

for n in x:
    y.append(heff_n_expectation_values_1d(n))
    z.append(heff_n_expectation_values_2d(n))

fig, ax = plt.subplots()


ax.plot(x, y, label="1D")
ax.plot(x, z, label="2D")

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Number of circuits")
ax.set_yscale("log")
ax.legend()
edm.save_figure(fig)
plt.show()
