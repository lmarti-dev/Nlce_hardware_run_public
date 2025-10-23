from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import Operator
import numpy as np
from qcomp_nlce.utils.hamiltonian import from_bitstring
from typing import Union
from qiskit.circuit import CircuitInstruction


def circuit_instruction_to_circuit(ci: type[CircuitInstruction] | list, qubits=None):
    if isinstance(ci, list):
        if qubits is None:
            qregs = [c.qubits[0]._register for c in ci]
            m = np.argmax([r.size for r in qregs])
            qubits = qregs[m]
        circ = QuantumCircuit(qubits)
        for ci_inst in ci:
            inds = [q._index for q in ci_inst.qubits]
            circ.append(ci_inst.operation, inds)
    else:
        if qubits is None:
            qubits = ci.qubits[0]._register
        circ = QuantumCircuit(qubits)
        inds = [q._index for q in ci.qubits]
        circ.append(ci.operation, inds)
    return circ


def apply_parameters(circuit: QuantumCircuit, parameters: np.ndarray) -> QuantumCircuit:
    options = {}
    if parameters is not None:
        param_dict = {k: v for k, v in zip(circuit.parameters, parameters)}
        options["parameter_map"] = param_dict
        return circuit.assign_parameters(param_dict)
    else:
        return circuit


def to_controlled_gate(circuit: QuantumCircuit, parameters: np.ndarray) -> Gate:
    circuit = apply_parameters(circuit, parameters)
    gate = circuit.to_gate().control(1)
    return gate


def parameters_dict(
    circuit: Union[QuantumCircuit, list[QuantumCircuit]], parameters: np.ndarray
) -> dict:
    if circuit.parameters is None or parameters is None:
        return {}
    return {k: v for k, v in zip(circuit.parameters, parameters)}


def isa_circuit_param_dict(
    new_circuit: QuantumCircuit, old_circuit: QuantumCircuit, parameters: np.ndarray
) -> dict:
    param_dict = parameters_dict(old_circuit, parameters)
    return {k: param_dict[k] for k in new_circuit.parameters}


def circuit_to_matrix(circuit: QuantumCircuit) -> np.ndarray:
    return Operator(circuit).to_matrix()


def circuit_matrix_elements(circuit: QuantumCircuit, states: list[str]) -> np.ndarray:
    mat = circuit_to_matrix(circuit)
    sl = [
        from_bitstring(state, n_qubits=circuit.num_qubits, right_to_left=True)
        for state in states
    ]
    return mat[np.ix_(sl, sl)]
