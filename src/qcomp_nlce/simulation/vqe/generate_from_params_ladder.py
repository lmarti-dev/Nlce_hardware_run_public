
# ── USER INPUT ────────────────────────────────────────────────────────────────
J = 0.2                # Ising coupling (overridden by file-parsed value)
H = 1                    # transverse field strength
HL = 0.0                 # longitudinal field strength
HAS_LONGITUDINAL = False # whether the circuit has longitudinal field rotations
# ──────────────────────────────────────────────────────────────────────────────

import re
import sys
import numpy as np
from pathlib import Path

from qiskit import QuantumCircuit
from qcomp_nlce.utils import (
    get_ground_energy,
    save_optimized_vqe,
    get_optimal_n_layers,
)
from qcomp_nlce.utils.hamiltonian import TFIM_ladder

JULIA_DATA_DIR = Path(__file__).parent.parent / "julia_data" / "TFIM_ladder"  # simulation/julia_data/TFIM_ladder


def parse_filename(filename: str) -> dict:
    """Parse a Julia ladder data filename into its components.

    Examples:
        gs_2x2_j0p328.csv   -> type=gs, rows=2, cols=2, J=0.328
        1qp_3x2_j0p2.csv    -> type=1qp, rows=3, cols=2, J=0.2
    """
    stem = Path(filename).stem  # strip .csv

    match = re.match(r"^(gs|1qp)_(\d+)x(\d+)_j([0-9p]+)$", stem)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")

    state_type = match.group(1)
    rows = int(match.group(2))
    cols = int(match.group(3))
    J = float(match.group(4).replace("p", "."))

    return {"type": state_type, "rows": rows, "cols": cols, "J": J}


def load_params(filepath: Path) -> np.ndarray:
    """Load parameters from a single-row CSV file."""
    return np.loadtxt(filepath, delimiter=",")


def get_ladder_bonds(m: int, n: int = 2):
    """Return the list of (i, j) bonds for an m x n lattice.

    Matches Julia's bond_coordinates + graph convention:
      - Sites are indexed column-major: site = i + j*m
        where i = 0..m-1 (rung index), j = 0..n-1 (leg index).
      - Bonds are all pairs (a, b) with a < b and Euclidean distance == 1,
        iterated in order of increasing a, then increasing b.

    For a ladder (n=2), m = n_rungs.
    """
    total = m * n
    # Column-major coordinates: site s -> (s % m, s // m)
    coords = [(s % m, s // m) for s in range(total)]
    bonds = []
    for a in range(total):
        for b in range(a + 1, total):
            dx = coords[b][0] - coords[a][0]
            dy = coords[b][1] - coords[a][1]
            if dx * dx + dy * dy == 1:
                bonds.append((a, b))
    return bonds


def build_hva_circuit_from_julia_params_ladder(
    n_qubits_per_chain: int, n_layers: int, params: np.ndarray,
    has_longitudinal_field: bool = False
) -> QuantumCircuit:
    """Build HVA circuit from Julia-optimized parameters for a ladder geometry.

    Julia uses exp(-i * theta * P) convention, while Qiskit's RXX/RX/RZ
    use exp(-i * theta/2 * P). Therefore we multiply Julia params by 2.

    Parameter structure per layer (matching Julia's Gates_tl order):
    - XX rotations for all ladder bonds (chain1, chain2, rungs)
    - X rotations (only if has_longitudinal_field)
    - Z rotations for all qubits
    """
    total_qubits = 2 * n_qubits_per_chain
    circuit = QuantumCircuit(total_qubits)

    bonds = get_ladder_bonds(n_qubits_per_chain, 2)
    n_xx = len(bonds)
    n_x = total_qubits if has_longitudinal_field else 0
    n_z = total_qubits
    params_per_layer = n_xx + n_x + n_z

    for layer in range(n_layers):
        base = layer * params_per_layer
        offset = 0
        # XX rotations on all ladder bonds (scaled by 2 for Qiskit convention)
        for bond_idx, (qi, qj) in enumerate(bonds):
            circuit.rxx(2 * params[base + offset + bond_idx], qi, qj)
        offset += n_xx
        # X longitudinal field rotations (scaled by 2)
        if has_longitudinal_field:
            for qubit in range(total_qubits):
                circuit.rx(2 * params[base + offset + qubit], qubit)
            offset += n_x
        # Z transverse field rotations (scaled by 2)
        for qubit in range(total_qubits):
            circuit.rz(2 * params[base + offset + qubit], qubit)

    return circuit


def generate_gs(n_qubits_per_chain: int, n_layers: int, J: float, h: float, hl: float,
                has_longitudinal_field: bool, params: np.ndarray):
    """Generate and save ground state VQE data for ladder geometry."""
    hamiltonian = TFIM_ladder(n_qubits_per_chain, h, J)
    total_qubits = hamiltonian.num_qubits

    circuit = build_hva_circuit_from_julia_params_ladder(
        n_qubits_per_chain, n_layers, params,
        has_longitudinal_field=has_longitudinal_field)
    ground_energy = get_ground_energy(hamiltonian)

    bonds = get_ladder_bonds(n_qubits_per_chain, 2)
    n_xx = len(bonds)
    n_x = total_qubits if has_longitudinal_field else 0
    n_z = total_qubits
    expected_params = n_layers * (n_xx + n_x + n_z)
    assert len(params) == expected_params, (
        f"Parameter count mismatch: file has {len(params)}, "
        f"circuit expects {expected_params}"
    )

    weights = [params]

    dirname = f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl_ladder"
    save_path = f"{dirname}/vqe/{n_qubits_per_chain}x2q_{n_layers}l"

    print(f"  Saving to res_sim/{save_path}/")
    save_optimized_vqe(save_path, hamiltonian, circuit, weights, ground_energy, dry_run=False)


def generate_elem(n_qubits_per_chain: int, n_layers: int, J: float, h: float, hl: float,
                  has_longitudinal_field: bool, params: np.ndarray):
    """Generate and save elementary (1qp) VQE data for ladder geometry."""
    ham_j, ham_h = TFIM_ladder(n_qubits_per_chain, h, J, split_parts=True,
                               )
    hamiltonian = ham_j + ham_h
    total_qubits = hamiltonian.num_qubits

    circuit = build_hva_circuit_from_julia_params_ladder(
        n_qubits_per_chain, n_layers, params,
        has_longitudinal_field=has_longitudinal_field)
    ground_energy = get_ground_energy(hamiltonian)

    bonds = get_ladder_bonds(n_qubits_per_chain, 2)
    n_xx = len(bonds)
    n_x = total_qubits if has_longitudinal_field else 0
    n_z = total_qubits
    expected_params = n_layers * (n_xx + n_x + n_z)
    assert len(params) == expected_params, (
        f"Parameter count mismatch: file has {len(params)}, "
        f"circuit expects {expected_params}"
    )

    weights = [params]

    dirname = f"elem_vqe_{int(J*100)}j_{int(hl*100)}hl_ladder"
    save_path = f"{dirname}/vqe/cost_offdiag/{n_qubits_per_chain}x2q_{n_layers}l"

    print(f"  Saving to res_sim/{save_path}/")
    save_optimized_vqe(save_path, hamiltonian, circuit, weights, ground_energy, dry_run=False)


def main(data_dir: Path = None):
    if data_dir is None:
        data_dir = JULIA_DATA_DIR

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} files in {data_dir}\n")

    for i, filepath in enumerate(csv_files):
        cfg = parse_filename(filepath.name)
        params = load_params(filepath)

        n_qubits_per_chain = cfg["rows"]
        total_qubits = 2 * n_qubits_per_chain
        n_layers = get_optimal_n_layers(total_qubits)

        print(f"[{i+1}/{len(csv_files)}] {filepath.name} -> "
              f"type={cfg['type']}, dims={cfg['rows']}x{cfg['cols']}, "
              f"total_qubits={total_qubits}, layers={n_layers}, "
              f"J={cfg['J']}, H={H}, hl={HL}, "
              f"longitudinal={HAS_LONGITUDINAL}, n_params={len(params)}")

        if cfg["type"] == "gs":
            generate_gs(n_qubits_per_chain, n_layers, cfg["J"], H, HL,
                        HAS_LONGITUDINAL, params)
        elif cfg["type"] == "1qp":
            generate_elem(n_qubits_per_chain, n_layers, cfg["J"], H, HL,
                          HAS_LONGITUDINAL, params)
        else:
            print(f"  WARNING: unknown type '{cfg['type']}', skipping")

    print("\nDone.")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(data_dir)
