import numpy as np

from qcomp_nlce.utils import (
    get_ground_energy,
    get_tfim_sweep_circuit,
    get_tfim_ham,
    get_circuit_energy,
    get_min_gap,
    save_sweep,
)

for n_qubits in range(2, 11):
    ham_j, ham_h = get_tfim_ham(n_qubits, 1, 0.7, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    n_layers = int(np.ceil(hamiltonian.num_qubits / 2))

    min_gap = get_min_gap(hamiltonian)

    circuit = get_tfim_sweep_circuit(
        ham_j, ham_h, total_time=1 / (min_gap**3), n_steps=5 * hamiltonian.num_qubits
    )
    # circuit = get_custom_vqe(4, n_layers=2)
    ground_energy = get_ground_energy(hamiltonian)

    print(f"The circuit depth is {circuit.depth()} for {hamiltonian.num_qubits} qubits")

    final_energy = get_circuit_energy(hamiltonian, circuit)
    print(
        f"final energy error is {np.abs(ground_energy-final_energy)/np.abs(ground_energy)}"
    )

    save_sweep(
        f"tfim_{hamiltonian.num_qubits}q_{n_layers}l_sweep",
        hamiltonian,
        circuit,
        ground_energy,
    )
