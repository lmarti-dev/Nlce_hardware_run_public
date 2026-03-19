from src.qcomp_nlce.utils import (
    TFIM,
    get_min_gap,
    get_tfim_sweep_circuit,
    circuit_to_matrix,
    compile_circuit_for_backend,
    to_controlled_gate,
    get_appropriate_fake_backend,
    sweep_time_from_steps,
)

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

n_qubits = 5
J = 1

ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
hamiltonian = ham_j + ham_h
min_gap = get_min_gap(hamiltonian)
# the most important thing is to have enough steps!!!
# the total time can be low-ish

n_steps = 2


total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
sweep_circuit = get_tfim_sweep_circuit(
    ham_h,
    ham_j + ham_h,
    total_time=total_time,
    n_steps=n_steps,
)

controlled_sweep = QuantumCircuit(n_qubits + 1)
controlled_sweep.append(to_controlled_gate(sweep_circuit, None), range(n_qubits + 1))


controlled_unitaryg = UnitaryGate(circuit_to_matrix(controlled_sweep))
unitaryg = UnitaryGate(circuit_to_matrix(sweep_circuit))


controlled_compiled_sweep = QuantumCircuit(n_qubits + 1)
controlled_compiled_sweep.append(controlled_unitaryg, range(n_qubits + 1))


compiled_sweep = QuantumCircuit(n_qubits)
compiled_sweep.append(unitaryg, range(n_qubits))

backend = get_appropriate_fake_backend(n_qubits + 1, "IBM")

print(f"backend: {backend.name}")
print("Sweep circuit")
compile_circuit_for_backend(sweep_circuit, backend)
print("Sweep to matrix circuit")
compile_circuit_for_backend(compiled_sweep, backend)
print("Controlled sweep circuit")
compile_circuit_for_backend(controlled_sweep, backend)
print("Controlled sweep to matrix circuit")
compile_circuit_for_backend(controlled_compiled_sweep, backend)
