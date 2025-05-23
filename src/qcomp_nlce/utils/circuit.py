from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
import numpy as np


def apply_parameters(circuit: QuantumCircuit, parameters: np.ndarray) -> QuantumCircuit:
    options = {}
    if parameters is not None:
        param_dict = {k: v for k, v in zip(circuit.parameters, parameters)}
        options["parameter_map"] = param_dict
    return circuit.assign_parameters(param_dict)


def to_controlled_gate(circuit: QuantumCircuit, parameters: np.ndarray) -> Gate:
    circuit = apply_parameters(circuit, parameters)
    gate = circuit.to_gate().control(1)
    return gate


def parameters_dict(circuit: QuantumCircuit, parameters: np.ndarray) -> dict:
    return {k: v for k, v in zip(circuit.parameters, parameters)}


def isa_circuit_param_dict(
    new_circuit: QuantumCircuit, old_circuit: QuantumCircuit, parameters: np.ndarray
) -> dict:
    param_dict = parameters_dict(old_circuit, parameters)
    return {k: param_dict[k] for k in new_circuit.parameters}
