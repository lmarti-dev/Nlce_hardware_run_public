import numpy as np
from uuid import uuid4

from qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
)


def train_and_save(n_qubits: int, J: float, dirname: str, dry_run: bool):
    periodic = False
    h = 1
    hamiltonian = TFIM(n_qubits, h, J, periodic=periodic)
    n_layers = int(np.ceil(hamiltonian.num_qubits / 2))

    circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)
    # circuit = get_custom_vqe(4, n_layers=2)
    ground_energy = get_ground_energy(hamiltonian)

    print(f"The circuit depth is {circuit.depth()}")
    print(f"The number of parameters is {len(circuit.parameters)}")
    energy_values, params = optimize_circuit(
        circuit,
        hamiltonian,
        initial_params=np.random.rand(len(circuit.parameters)),
    )
    print(
        f"final energy error is {np.abs(ground_energy-energy_values[-1])/np.abs(ground_energy)}"
    )

    save_optimized_vqe(
        f"{dirname}/vqe/{n_qubits}q_{n_layers}l",
        hamiltonian,
        circuit,
        params,
        ground_energy,
        dry_run,
    )


dry_run = False
J_range = (1.0,)
dirname = "gs_vqe_1j"
for n_qubits in (4, 5):
    for J in J_range:
        train_and_save(n_qubits, J, dirname, dry_run)
