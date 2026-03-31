from qiskit_aer import AerSimulator, noise
from typing import Literal

SIM_METHODS = Literal[
    "automatic",
    "statevector",
    "density_matrix",
    "stabilizer",
    "matrix_product_state",
    "extended_stabilizer",
    "unitary",
    "superop",
    "tensor_network",
]


def depolarizing_backend(
    _1q: float = 1, _2q: float = 1, method: SIM_METHODS = "statevector"
):
    noise_model = noise.NoiseModel(basis_gates=["r", "rz", "rxx"])
    noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(_1q, 1), ["r"])
    noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(_2q, 2), ["rxx"])

    simulator = AerSimulator(method=method, noise_model=noise_model)
    return simulator
