from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps

c = QuantumCircuit(1)
c.r(1, 1, 0)

xx = dumps(c)

print(xx)
