from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT = (
    Path(__file__).resolve().parents[1] / "res_sim" / "early_comp" / "vqe"
)
DEFAULT_OUTPUT = RESULTS_ROOT / "ground_state_energy_vs_J.png"
DEFAULT_TABLE = RESULTS_ROOT / "ground_state_energy_vs_J.csv"


def infer_j_from_terms(pauli_terms: Iterable[Tuple[str, complex]]) -> float:
    """Infer J from the XX terms of a TFIM Hamiltonian."""
    j_samples: List[float] = []
    for pauli, coeff in pauli_terms:
        label = str(pauli)
        if label.count("X") == 2 and set(label.replace("X", "")) <= {"I"}:
            j_samples.append(-float(np.real(coeff)))
    if not j_samples:
        raise ValueError("No XX terms found to infer J.")
    return float(np.mean(j_samples))


def load_run(run_dir: Path) -> Tuple[int, float, float]:
    ground_energy = float(np.load(run_dir / "ground_energy.npy"))
    ham_terms = np.load(run_dir / "hamiltonian.npy", allow_pickle=True)
    pauli_terms = []
    for term in ham_terms:
        if isinstance(term, (list, tuple)):
            label, coeff = term
        elif hasattr(term, "__len__"):
            label, coeff = term[0], term[1]
        else:
            label, coeff = term.item()  # type: ignore[assignment]
        pauli_terms.append((str(label), complex(coeff)))
    n_qubits = len(pauli_terms[0][0])
    j_value = infer_j_from_terms(pauli_terms)
    return n_qubits, j_value, ground_energy


def collect_results(
    result_dirs: Iterable[Path],
) -> Dict[int, List[Tuple[float, float, str]]]:
    data: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)
    for run_dir in result_dirs:
        if not run_dir.is_dir():
            continue
        try:
            n_qubits, j_value, energy = load_run(run_dir)
        except (FileNotFoundError, ValueError, IndexError) as exc:
            print(f"Skipping {run_dir.name}: {exc}")
            continue
        data[n_qubits].append((j_value, energy, run_dir.name))
    for runs in data.values():
        runs.sort(key=lambda item: item[0])
    return data


def save_table(
    data: Dict[int, List[Tuple[float, float, str]]], table_path: Path
) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["run_id,n_qubits,J,ground_energy"]
    for n_qubits, entries in data.items():
        for j_value, energy, run_id in entries:
            rows.append(f"{run_id},{n_qubits},{j_value:.8f},{energy:.8f}")
    table_path.write_text("\n".join(rows))


def plot_data(
    data: Dict[int, List[Tuple[float, float, str]]], output: Path, show: bool = False
) -> None:
    fig, ax = plt.subplots()
    for n_qubits, entries in sorted(data.items()):
        j_vals = [item[0] for item in entries]
        energies = [item[1] for item in entries]
        ax.plot(
            j_vals,
            energies,
            marker="o",
            label=f"N={n_qubits}",
        )
    ax.set_xlabel("J")
    ax.set_ylabel("Ground-state energy")
    ax.set_title("TFIM ground-state energy vs J")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ground-state energy vs J from saved VQE runs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Directory that contains the VQE result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the generated figure.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help="CSV path for the extracted energies.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"Results directory {args.results_root} does not exist.")

    run_dirs = sorted(args.results_root.iterdir())
    data = collect_results(run_dirs)
    if not data:
        raise SystemExit("No valid result folders found.")

    plot_data(data, args.output, show=args.show)
    save_table(data, args.table)
    print(f"Saved figure to {args.output}")
    print(f"Saved table to {args.table}")


if __name__ == "__main__":
    main()
