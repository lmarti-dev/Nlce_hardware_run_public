from qcomp_nlce.utils import swap_test, hadamard_test
from qiskit import QuantumCircuit

n_shots = 100000

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.h(1)


swap_res = swap_test(circuit, n_shots=n_shots)
hadam_res = hadamard_test(circuit, n_shots=n_shots)

# expected f"〈ψ|U|ψ〉²: .25, Re〈ψ|U|ψ〉: .5"
print(f"〈ψ|U|ψ〉²: {swap_res}, Re〈ψ|U|ψ〉: {hadam_res}")
