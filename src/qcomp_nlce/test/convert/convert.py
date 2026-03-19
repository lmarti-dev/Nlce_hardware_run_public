from src.qcomp_nlce.utils import (
    TFIM,
    save_optimized_vqe,
    get_hva_vqe,
    apply_parameters,
    get_ground_energy,
    circmat_observables,
)
import numpy as np

from pathlib import Path
import json

HOME = Path(__file__).parent


def load_json(fpath: Path):
    with open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read())
    return jobj


if __name__ == "__main__":
    d = [
        {
            "filename": "all_gs_params7.json",
            "n_qubits": 7,
            "hl": 0,
            "ladder": False,
            "periodic": False,
        },
        {
            "filename": "all_gs_params8.json",
            "n_qubits": 8,
            "hl": 0,
            "ladder": False,
            "periodic": False,
        },
    ]

    dry_run = True

    for item in d:
        n_qubits = item["n_qubits"]
        ladder = item["ladder"]
        hl = item["hl"]
        periodic = item["periodic"]
        filename = item["filename"]

        all_params = load_json(Path(HOME, filename))
        js = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for ind, params in enumerate(all_params):
            J = js[ind]

            n_layers = n_qubits // 2

            dirname = f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl" + "_ladder" * int(ladder)
            h = 1
            hamiltonian = TFIM(
                n_qubits, h, J, periodic=periodic, longitudinal_field=hl, ladder=ladder
            )
            ground_energy = get_ground_energy(hamiltonian)
            circuit = get_hva_vqe(hamiltonian, n_layers, True)
            print(f"circuit params: {len(circuit.parameters)} param vec: {len(params)}")
            vqe_gs = circmat_observables(circuit, hamiltonian.to_matrix(), params)
            print(f"j={J}, {n_qubits} qubits, err: {ground_energy-vqe_gs}")

            save_optimized_vqe(
                f"{dirname}/vqe/{n_qubits}q_{n_layers}l",
                hamiltonian,
                circuit,
                np.real(params),
                ground_energy,
                dry_run,
            )
