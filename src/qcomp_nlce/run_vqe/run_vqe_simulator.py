import numpy as np

from qcomp_nlce.utils import (
    get_custom_vqe,
    get_ground_energy,
    get_hva_vqe,
    get_tfim_ham,
    optimize_circuit,
    save_optimized_vqe,
)

hamiltonian = get_tfim_ham(4, 1, 0.7, periodic=False)
n_layers = int(np.floor(hamiltonian.num_qubits) / 2)

circuit = get_hva_vqe(hamiltonian, n_layers=n_layers)
# circuit = get_custom_vqe(4, n_layers=2)
ground_energy = get_ground_energy(hamiltonian)


print(f"The circuit depth is {circuit.depth()}")
print(f"The number of parameters is {len(circuit.parameters)}")
energy_values, params = optimize_circuit(circuit, hamiltonian)
print(
    f"final energy error is {np.abs(ground_energy-energy_values[-1])/np.abs(ground_energy)}"
)


save_optimized_vqe(
    f"{hamiltonian.num_qubits}q_tfim_{n_layers}_layers_hva",
    hamiltonian,
    circuit,
    params,
    ground_energy,
)
