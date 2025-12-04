from qcomp_nlce.utils import amount_of_vals_needed
import matplotlib.pyplot as plt

x = range(3, 100)
y = []

for n in x:
    y.append(amount_of_vals_needed(n))

fig, ax = plt.subplots()


ax.plot(x, y)

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Number of expectation values for pCAT")
ax.set_yscale("log")

plt.show()
