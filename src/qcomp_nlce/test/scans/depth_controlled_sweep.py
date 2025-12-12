from qiskit import QuantumCircuit


from qcomp_nlce.utils import (
    get_tfim_sweep_circuit,
    TFIM,
    get_min_gap,
    get_appropriate_fake_backend,
    to_controlled_gate,
)

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


if __name__ == "__main__":
    n_qubits = 10

    ham_j, ham_h = TFIM(n_qubits, 1, 1, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = 1 / (min_gap**2)
    n_steps = 10 * hamiltonian.num_qubits
    mult = 1

    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=(mult * total_time),
        n_steps=int(mult * n_steps),
    )

    backend = get_appropriate_fake_backend(circuit.num_qubits, "IBM")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)

    controlled_circuit = QuantumCircuit(circuit.num_qubits + 1)
    controlled_gate = to_controlled_gate(circuit, None)
    controlled_circuit.append(controlled_gate, controlled_circuit.qubits)
    isa_controlled_circuit = pm.run(controlled_circuit)

    print(
        f"Simple: {isa_circuit.depth()}, controlled: {isa_controlled_circuit.depth()}"
    )

    ...
