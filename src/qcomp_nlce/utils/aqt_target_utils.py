from qiskit import QuantumCircuit, transpile

try:
    from qiskit_aqt_provider import AQTProvider
except Exception:

    class AQTProvider: ...


from qiskit.transpiler import Target, CouplingMap
from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping


def make_custom_target(base: Target, basis_gates: list[str]):
    name_map = get_standard_gate_name_mapping()
    custom_target = Target()
    # This gets a coupling map from the union of all 2q links, regardless of
    # active gate and directionality.
    num_qubits = len(base.physical_qubits)
    base_all_2q_coupling = CouplingMap.from_full(num_qubits)
    for gate_name in basis_gates:
        gate = name_map[gate_name]
        if gate.num_qubits == 1:
            # Assume that any 1q gate is available as an ideal gate on all
            # qubits. Throws away noise information.
            custom_target.add_instruction(
                gate, {(qubit,): None for qubit in range(base.num_qubits)}
            )
        elif gate.num_qubits == 2:
            # Assume that any 2q gate is availble bidirectionally on all
            # pairs of qubits that have _any_ 2q gate defined on them in
            # the base.  Throws away noise information.
            custom_target.add_instruction(
                gate, {qargs: None for qargs in base_all_2q_coupling}
            )
        else:
            raise ValueError("don't know what to do with this")
    if "measure" not in basis_gates and "measure" in base:
        # Assume that the same individual qubits are measurable.
        custom_target.add_instruction(
            Measure(),
            {qargs: None for qargs in base.qargs_for_operation_name("measure")},
        )
    return custom_target


def custom_aqt_target():
    basis_gates = ["rz", "r", "rxx", "measure"]

    backend = AQTProvider("ANY_TOKEN_WORKS_FOR_OFFLINE_SIM").get_backend(
        "offline_simulator_noise"
    )

    return make_custom_target(backend.target, basis_gates)
