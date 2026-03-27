from qiskit_aer import AerSimulator, noise


def depolarizing_backend(_1q: float = 1, _2q: float = 1):
    noise_model = noise.NoiseModel(basis_gates=["r", "rz", "rxx"])
    noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(_1q, 1), ["r"])
    noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(_2q, 2), ["rxx"])

    simulator = AerSimulator(method="statevector", noise_model=noise_model)
    return simulator
