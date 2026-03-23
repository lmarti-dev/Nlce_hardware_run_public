import numpy as np
from uuid import uuid4

from src.qcomp_nlce.utils import (
    get_ground_energy,
    get_hva_vqe,
    TFIM,
    optimize_circuit,
    save_optimized_vqe,
    get_optimal_n_layers,
)


def train_and_save(
    n_qubits: int,
    J: float,
    hl: float,
    dirname: str,
    dry_run: bool,
    ladder: bool = False,
):
    periodic = False
    h = 1
    hamiltonian = TFIM(
        n_qubits, h, J, periodic=periodic, longitudinal_field=hl, ladder=ladder
    )
    n_qubits = hamiltonian.num_qubits
    print(f"{n_qubits} qubits, J/h={J}, ladder: {ladder}, hl: {hl:.4f}")
    n_layers = get_optimal_n_layers(hamiltonian.num_qubits)

    circuit = get_hva_vqe(hamiltonian, n_layers=n_layers, unique_weights=True)
    ground_energy = get_ground_energy(hamiltonian)

    print(f"The circuit depth is {circuit.depth()}")
    print(f"The number of parameters is {len(circuit.parameters)}")
    energy_values, params = optimize_circuit(
        circuit,
        hamiltonian,
        initial_params=0.1 * np.ones((len(circuit.parameters),)),
        target=ground_energy,
    )
    print(
        f"final energy error is {np.abs(ground_energy-energy_values[-1])/np.abs(ground_energy)}"
    )

    save_optimized_vqe(
        f"{dirname}/vqe/{n_qubits}q_{n_layers}l",
        hamiltonian,
        circuit,
        np.real(params),
        ground_energy,
        dry_run,
    )


if __name__ == "__main__":
    dry_run = False
    J_range = np.linspace(0.1, 1, 10)
    hl = 0.0
    ladder = False
    for n_qubits in (9, 10):
        for J in J_range:
            dirname = f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl" + "_ladder" * int(ladder)
            train_and_save(n_qubits, J, hl, dirname, dry_run, ladder)
