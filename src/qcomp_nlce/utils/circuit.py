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
