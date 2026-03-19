from src.qcomp_nlce.utils import (
    heff_n_expectation_values_1d,
    heff_n_expectation_values_2d,
)

import matplotlib.pyplot as plt
from fauplotstyle import style
from data_manager import ExperimentDataManager


dry_run = False

edm = ExperimentDataManager(
    "expectation_values_scaling", project="nlce", dry_run=dry_run
)


style()
x = range(3, 100)
y = []
l = []
z = []

for n in x:
    y.append(heff_n_expectation_values_1d(n))
    # l.append(heff_n_expectation_values_ladder(n))
    z.append(heff_n_expectation_values_2d(n))

fig, ax = plt.subplots()


ax.plot(x, y, label="Chain")
# ax.plot(x, l, label="Ladder")
ax.plot(x, z, label="Square lattice")

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Number of circuits")
ax.set_yscale("log")
ax.legend()
edm.save_figure(fig, filename="n_circuits", add_timestamp=False)
plt.show()
