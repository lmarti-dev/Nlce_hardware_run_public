"""
zne_dispersion.py
=================
Zero-Noise Extrapolation (ZNE) for 1qp dispersion via unitary folding.

Method:
  1. For each noise scale factor λ ∈ {1, 3, 5, ...}, fold the VQE circuits
     by appending (U†U) pairs: λ=1 is the original circuit, λ=3 appends one
     U†U pair, λ=5 appends two, etc.
  2. Run the full H_eff pipeline at each noise level on the fake backend,
     averaging over multiple repetitions to reduce shot noise.
  3. Extrapolate each H_eff matrix element to λ→0 using polynomial
     (Richardson) extrapolation (linear by default).
  4. Build the zero-noise H_eff and compute dispersion.

Usage:
    from qcomp_nlce.simulation.vqe.zne_dispersion import zne_generate, zne_plot_single, zne_plot_nlce
    zne_generate(J=1.0, hl=0.0, cluster_pairs=[(5, 4)])
    zne_plot_single(N=5, J=1.0, hl=0.0)
    zne_plot_nlce(cluster_pairs=[(5, 4)], J=1.0, hl=0.0)
"""

import numpy as np
from copy import deepcopy
from pathlib import Path
from qiskit import QuantumCircuit

from qcomp_nlce.utils.save import load_optimized_vqe, get_save_directory, baptize_circuit
from qcomp_nlce.utils.circuit import apply_parameters
from qcomp_nlce.utils.hamiltonian import TFIM, get_ground_energy, get_tfim_ratio, HAM_NAME_VAR
from qcomp_nlce.utils.effective_hamiltonian import H_effective, H_effective_ed
from qcomp_nlce.utils.vqe import get_optimal_n_layers
from qcomp_nlce.utils.workflow_tfim_1d import get_heff_values
from qcomp_nlce.utils.dispersion import cluster_dispersion, dispersion_1d_thermodynamic


# ── Paths ────────────────────────────────────────────────────────────────────

RES_SIM = Path(__file__).parent.parent.parent / "res_sim"
DATA_DIR = RES_SIM / "dispersion_data"
FIG_DIR = RES_SIM / "figures"


def _zne_cluster_dir(N: int, J: float, hl: float) -> Path:
    d = DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"N{N}" / "zne"
    d.mkdir(exist_ok=True, parents=True)
    return d


# ── Unitary folding ─────────────────────────────────────────────────────────

def fold_circuit(circuit: QuantumCircuit, n_folds: int) -> QuantumCircuit:
    """
    Apply global unitary folding: C → C (C† C)^n_folds.

    The effective noise scale factor is λ = 1 + 2 * n_folds.
    n_folds=0 returns the original circuit (λ=1).
    n_folds=1 returns C C† C (λ=3).
    n_folds=2 returns C C† C C† C (λ=5).
    """
    if n_folds == 0:
        return circuit.copy()

    folded = circuit.copy()
    for _ in range(n_folds):
        folded.compose(circuit.inverse(), inplace=True)
        folded.compose(circuit, inplace=True)
    return folded


def noise_scale_factors(n_folds_list: list[int]) -> np.ndarray:
    """Convert list of fold counts to noise scale factors λ = 1 + 2*n."""
    return np.array([1 + 2 * n for n in n_folds_list], dtype=float)


# ── Richardson extrapolation ────────────────────────────────────────────────

def richardson_extrapolate(
    scale_factors: np.ndarray,
    values: np.ndarray,
    order: int = None,
) -> complex:
    """
    Richardson extrapolation to zero noise (λ→0).

    Uses polynomial fit of degree = order (default: 1 = linear).
    Evaluates the fitted polynomial at λ=0.
    """
    if order is None:
        order = 1
    order = min(order, len(scale_factors) - 1)

    if np.iscomplexobj(values):
        coeffs_r = np.polyfit(scale_factors, values.real, order)
        coeffs_i = np.polyfit(scale_factors, values.imag, order)
        return np.polyval(coeffs_r, 0) + 1j * np.polyval(coeffs_i, 0)
    else:
        coeffs = np.polyfit(scale_factors, values, order)
        return np.polyval(coeffs, 0)


def extrapolate_matrix(
    scale_factors: np.ndarray,
    matrices: list[np.ndarray],
    order: int = None,
) -> np.ndarray:
    """Extrapolate each element of a matrix independently."""
    shape = matrices[0].shape
    result = np.zeros(shape, dtype=complex)
    for i in range(shape[0]):
        for j in range(shape[1]):
            vals = np.array([m[i, j] for m in matrices])
            result[i, j] = richardson_extrapolate(scale_factors, vals, order)
    return result


# ── VQE data loader ─────────────────────────────────────────────────────────

def _load_vqe_data(N: int, J: float, hl: float) -> dict:
    n_layers = get_optimal_n_layers(N)
    gs_dir = str(get_save_directory(f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/{N}q_{n_layers}l"))
    elem_dir = str(get_save_directory(f"elem_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/cost_offdiag/{N}q_{n_layers}l"))

    ham_gs, gs_circuit, gs_weights, _ = load_optimized_vqe(gs_dir)
    ham_elem, elem_circuit, elem_weights, _ = load_optimized_vqe(elem_dir)

    gs_bound = apply_parameters(gs_circuit, gs_weights[-1])
    elem_bound = apply_parameters(elem_circuit, elem_weights[-1])

    return {
        "hamiltonian": ham_elem,
        "gs_circuit": gs_circuit,
        "gs_weights": gs_weights,
        "gs_bound": gs_bound,
        "elem_circuit": elem_circuit,
        "elem_weights": elem_weights,
        "elem_bound": elem_bound,
    }


# ── Compute H_eff at a given noise level ────────────────────────────────────

def compute_heff_folded(
    vqe: dict,
    n_folds: int,
    which_backend: str,
    provider_name: str,
    n_shots: int,
    which_hadamard_test: str,
    rep_index: int = 0,
) -> dict:
    """
    Compute H_eff with folded circuits at a given noise level.

    Returns dict with keys: heff, gs_energy, scale_factor, n_folds
    """
    hamiltonian = vqe["hamiltonian"]
    hamiltonian.__dict__[HAM_NAME_VAR] = "TFIM"

    # Fold both gs and elem circuits
    gs_folded = fold_circuit(vqe["gs_bound"], n_folds)
    elem_folded = fold_circuit(vqe["elem_bound"], n_folds)

    # Stamp metadata for caching
    n_qubits = hamiltonian.num_qubits
    ratio = get_tfim_ratio(hamiltonian)
    base_name = baptize_circuit(
        "dim1d", "vqe", n_qubits, ratio, provider_name, n_shots, which_hadamard_test
    )
    tmpname = f"{base_name}_{which_backend}_zne_fold{n_folds}_rep{rep_index}"

    gs_circ = deepcopy(gs_folded)
    elem_circ = deepcopy(elem_folded)
    gs_circ.metadata = {"tmpname": tmpname}
    elem_circ.metadata = {"tmpname": tmpname}

    scale = 1 + 2 * n_folds

    print(f"    ZNE fold={n_folds} (λ={scale}) rep={rep_index}: "
          f"gs depth {gs_folded.depth()}, elem depth {elem_folded.depth()}")

    result = get_heff_values(
        gs_circ, elem_circ, hamiltonian,
        which_backends=[which_backend],
        provider_name=provider_name,
        n_shots=n_shots,
        group_size=None,
        which_hadamard_test=which_hadamard_test,
    )

    return {
        "heff": result["heff"][which_backend],
        "overlap": result["eld"][which_backend]["values"],
        "expect": result["exd"][which_backend]["values"],
        "gs_energy": result["gsd"][which_backend]["energy"],
        "scale_factor": scale,
        "n_folds": n_folds,
    }


# ── Save/load ZNE data ─────────────────────────────────────────────────────

def save_zne_data(cdir: Path, data: dict):
    """Save all ZNE noise-level data and the extrapolated result."""
    np.savez(
        cdir / "zne_data.npz",
        scale_factors=data["scale_factors"],
        heff_extrapolated=data["heff_extrapolated"],
        n_reps=data["n_reps"],
        allow_pickle=True,
    )
    # Save per-level averaged H_eff matrices
    for i, sf in enumerate(data["scale_factors"]):
        np.save(cdir / f"H_eff_lambda{int(sf)}.npy", data["heff_per_level"][i])
    # Save per-level per-rep H_eff matrices
    for i, sf in enumerate(data["scale_factors"]):
        for r, heff in enumerate(data["heff_per_level_reps"][i]):
            np.save(cdir / f"H_eff_lambda{int(sf)}_rep{r}.npy", heff)
    # Save extrapolated
    np.save(cdir / "H_eff_ZNE.npy", data["heff_extrapolated"])


def load_zne_heff(N: int, J: float, hl: float) -> np.ndarray:
    """Load the ZNE-extrapolated H_eff for cluster N."""
    cdir = _zne_cluster_dir(N, J, hl)
    p = cdir / "H_eff_ZNE.npy"
    if p.exists():
        return np.load(p)
    return None


def load_zne_data(N: int, J: float, hl: float) -> dict:
    """Load all ZNE data for cluster N."""
    cdir = _zne_cluster_dir(N, J, hl)
    p = cdir / "zne_data.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    scale_factors = d["scale_factors"]
    n_reps = int(d["n_reps"]) if "n_reps" in d else 1
    heff_per_level = []
    heff_per_level_reps = []
    for sf in scale_factors:
        heff_per_level.append(np.load(cdir / f"H_eff_lambda{int(sf)}.npy"))
        reps = []
        for r in range(n_reps):
            rp = cdir / f"H_eff_lambda{int(sf)}_rep{r}.npy"
            if rp.exists():
                reps.append(np.load(rp))
        heff_per_level_reps.append(reps)
    return {
        "scale_factors": scale_factors,
        "heff_per_level": heff_per_level,
        "heff_per_level_reps": heff_per_level_reps,
        "heff_extrapolated": d["heff_extrapolated"],
        "n_reps": n_reps,
    }


# ── Main ZNE workflow ───────────────────────────────────────────────────────

def zne_generate(
    J: float = 1.0,
    hl: float = 0.0,
    cluster_pairs: list[tuple[int, int]] = None,
    provider_name: str = "AQT",
    n_shots: int = 200,
    which_hadamard_test: str = "stefan",
    which_backend: str = "fake_backend",
    n_folds_list: list[int] = None,
    n_repetitions: int = 10,
    extrapolation_order: int = None,
):
    """
    Run ZNE for each cluster size.

    Parameters
    ----------
    n_folds_list : list of fold counts. Default [0, 1, 2] -> lambda = [1, 3, 5].
    n_repetitions : number of repetitions per noise level (averaged before extrapolation).
    extrapolation_order : polynomial degree for Richardson. Default = 1 (linear).
    """
    if cluster_pairs is None:
        cluster_pairs = [(4, 3)]
    if n_folds_list is None:
        n_folds_list = [0, 1, 2]

    scale_factors = noise_scale_factors(n_folds_list)
    all_n = sorted(set(n for pair in cluster_pairs for n in pair))

    print(f"\n{'='*60}")
    print(f"ZNE Dispersion Data Generation")
    print(f"J={J}, hl={hl}, backend={which_backend}")
    print(f"Noise scale factors: {scale_factors}")
    print(f"Repetitions per level: {n_repetitions}")
    print(f"Extrapolation order: {extrapolation_order or 1} (linear)")
    print(f"{'='*60}")

    for N in all_n:
        print(f"\n--- N={N} ---")
        cdir = _zne_cluster_dir(N, J, hl)
        vqe = _load_vqe_data(N, J, hl)

        # Compute ED reference for diagnostics
        ham = TFIM(N, h=1, J=J, periodic=False, longitudinal_field=hl)
        heff_ed = H_effective_ed(ham)

        heff_per_level = []       # averaged H_eff at each lambda
        heff_per_level_reps = []  # list of per-rep H_eff at each lambda
        failed = False

        for n_folds in n_folds_list:
            scale = 1 + 2 * n_folds
            print(f"\n  lambda={scale} ({n_repetitions} reps):")
            reps = []
            for rep in range(n_repetitions):
                try:
                    result = compute_heff_folded(
                        vqe, n_folds, which_backend, provider_name,
                        n_shots, which_hadamard_test, rep_index=rep,
                    )
                    reps.append(result["heff"])
                except Exception as e:
                    print(f"    FAILED at fold={n_folds} rep={rep}: {e}")

            if not reps:
                print(f"  lambda={scale}: ALL reps failed, aborting N={N}")
                failed = True
                break

            # Average over reps
            avg = np.mean(reps, axis=0)
            dist = np.linalg.norm(avg - heff_ed)
            print(f"  lambda={scale}: {len(reps)}/{n_repetitions} reps, "
                  f"||H_eff - ED||_F = {dist:.4f}")

            heff_per_level.append(avg)
            heff_per_level_reps.append(reps)

        if failed:
            continue

        # Extrapolate
        heff_zne = extrapolate_matrix(scale_factors, heff_per_level, extrapolation_order)
        dist_zne = np.linalg.norm(heff_zne - heff_ed)
        print(f"\n  ZNE extrapolated: ||H_eff_ZNE - ED||_F = {dist_zne:.4f}")

        data = {
            "scale_factors": scale_factors,
            "heff_per_level": heff_per_level,
            "heff_per_level_reps": heff_per_level_reps,
            "heff_extrapolated": heff_zne,
            "n_reps": n_repetitions,
        }
        save_zne_data(cdir, data)
        print(f"  N={N} ZNE done -> {cdir}")

    print(f"\n{'='*60}")
    print("ZNE data generation complete.")


# ── Plotting ────────────────────────────────────────────────────────────────

def zne_plot_single(
    N: int, J: float, hl: float,
    save: bool = True,
    n_k_points: int = 20,
):
    """Plot single-cluster dispersion comparing ED, FB, and ZNE."""
    import matplotlib.pyplot as plt

    ks = np.linspace(0, np.pi, n_k_points)

    # ED
    ham = TFIM(N, h=1, J=J, periodic=False, longitudinal_field=hl)
    heff_ed = H_effective_ed(ham)
    disp_ed = [np.real(cluster_dispersion(N, heff_ed, k)) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ks, disp_ed, '--s', color='black', linewidth=1.5, markersize=4,
            label="ED", zorder=5)

    # Load FB data (unfolded, from regular pipeline)
    fb_dir = DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"N{N}"
    fb_heffs = []
    for i in range(1000):
        p = fb_dir / f"H_eff_fake_backend_rep{i}.npy"
        if not p.exists():
            break
        fb_heffs.append(np.load(p))

    if fb_heffs:
        all_disp = np.array([
            [np.real(cluster_dispersion(N, h, k)) for k in ks]
            for h in fb_heffs
        ])
        mean = np.mean(all_disp, axis=0)
        n_reps = len(all_disp)
        sem = np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else np.zeros_like(mean)
        ax.errorbar(ks, mean, yerr=sem, fmt='-^', color='tab:orange', markersize=4,
                    capsize=3, linewidth=1.5, label="Fake Backend", zorder=2)

    # ZNE
    heff_zne = load_zne_heff(N, J, hl)
    if heff_zne is not None:
        disp_zne = [np.real(cluster_dispersion(N, heff_zne, k)) for k in ks]
        ax.plot(ks, disp_zne, '-D', color='tab:purple', linewidth=2, markersize=5,
                label="ZNE", zorder=6)

    # Per-level mean curves (diagnostic)
    zne_data = load_zne_data(N, J, hl)
    if zne_data is not None:
        for i, (sf, heff) in enumerate(zip(zne_data["scale_factors"], zne_data["heff_per_level"])):
            disp = [np.real(cluster_dispersion(N, heff, k)) for k in ks]
            ax.plot(ks, disp, ':', color=f'C{i+3}', linewidth=1, alpha=0.5,
                    label=f"λ={int(sf)}")

    ax.set_xlabel("k")
    ax.set_ylabel("ω(k)")
    ax.set_title(f"ZNE 1qp Dispersion: N={N} (J={J}, hl={hl})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(exist_ok=True, parents=True)
        fname = FIG_DIR / f"zne_single_N{N}_j{int(J*100)}_hl{int(hl*100)}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


def zne_plot_nlce(
    cluster_pairs: list[tuple[int, int]],
    J: float, hl: float,
    save: bool = True,
    n_k_points: int = 20,
):
    """Plot NLCE dispersion comparing ED, FB, and ZNE."""
    import matplotlib.pyplot as plt

    ks = np.linspace(0, np.pi, n_k_points)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Exact analytic (only meaningful for NLCE, not single cluster)
    if np.isclose(hl, 0.0):
        from qcomp_nlce.utils.dispersion import exact_dispersion_1qp_1d
        ks_dense = np.linspace(0, np.pi, 200)
        ax.plot(ks_dense, exact_dispersion_1qp_1d(J, ks_dense), '-', color='gray',
                linewidth=2.5, label="Exact analytic", zorder=0)

    for n_big, n_small in cluster_pairs:
        # ED
        ham_b = TFIM(n_big, h=1, J=J, periodic=False, longitudinal_field=hl)
        ham_s = TFIM(n_small, h=1, J=J, periodic=False, longitudinal_field=hl)
        heff_ed_b = H_effective_ed(ham_b)
        heff_ed_s = H_effective_ed(ham_s)
        disp_ed = [dispersion_1d_thermodynamic(k, heff_ed_b, heff_ed_s) for k in ks]
        ax.plot(ks, disp_ed, '--s', color='black', linewidth=1.5, markersize=4,
                label="NLCE ED", zorder=5)

        # FB from regular pipeline
        fb_dir_b = DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"N{n_big}"
        fb_dir_s = DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"N{n_small}"
        fb_big, fb_small = [], []
        for i in range(1000):
            pb = fb_dir_b / f"H_eff_fake_backend_rep{i}.npy"
            ps = fb_dir_s / f"H_eff_fake_backend_rep{i}.npy"
            if not pb.exists() or not ps.exists():
                break
            fb_big.append(np.load(pb))
            fb_small.append(np.load(ps))

        if fb_big:
            n_reps = len(fb_big)
            all_disp = np.array([
                [dispersion_1d_thermodynamic(k, fb_big[r], fb_small[r]) for k in ks]
                for r in range(n_reps)
            ])
            mean = np.mean(all_disp, axis=0)
            sem = np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else np.zeros_like(mean)
            ax.errorbar(ks, mean, yerr=sem, fmt='-^', color='tab:orange', markersize=4,
                        capsize=3, linewidth=1.5, label="Fake Backend", zorder=2)

        # ZNE
        heff_zne_b = load_zne_heff(n_big, J, hl)
        heff_zne_s = load_zne_heff(n_small, J, hl)
        if heff_zne_b is not None and heff_zne_s is not None:
            disp_zne = [dispersion_1d_thermodynamic(k, heff_zne_b, heff_zne_s) for k in ks]
            ax.plot(ks, disp_zne, '-D', color='tab:purple', linewidth=2, markersize=5,
                    label="ZNE", zorder=6)

    ax.set_xlabel("k")
    ax.set_ylabel("ω(k)")
    ax.set_title(f"ZNE NLCE 1qp Dispersion (J={J}, hl={hl})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(exist_ok=True, parents=True)
        fname = FIG_DIR / f"zne_nlce_j{int(J*100)}_hl{int(hl*100)}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


# ── Diagnostic: extrapolation quality ───────────────────────────────────────

def zne_extrapolation_plot(
    N: int, J: float, hl: float,
    element: tuple[int, int] = (0, 0),
    save: bool = True,
):
    """
    Plot a single H_eff matrix element vs noise scale factor,
    showing the measured points and the Richardson fit.
    """
    import matplotlib.pyplot as plt

    zne_data = load_zne_data(N, J, hl)
    if zne_data is None:
        print(f"No ZNE data for N={N}")
        return

    i, j = element
    sf = zne_data["scale_factors"]
    vals = np.array([m[i, j] for m in zne_data["heff_per_level"]])
    extrap = zne_data["heff_extrapolated"][i, j]

    # ED reference
    ham = TFIM(N, h=1, J=J, periodic=False, longitudinal_field=hl)
    heff_ed = H_effective_ed(ham)
    ed_val = heff_ed[i, j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Per-rep scatter
    for li, (s, reps) in enumerate(zip(sf, zne_data["heff_per_level_reps"])):
        if reps:
            rep_vals = np.array([r[i, j] for r in reps])
            ax1.scatter([s] * len(rep_vals), rep_vals.real,
                        alpha=0.3, color='tab:orange', s=15, zorder=1)
            ax2.scatter([s] * len(rep_vals), rep_vals.imag,
                        alpha=0.3, color='tab:orange', s=15, zorder=1)

    # Real part
    ax1.plot(sf, vals.real, 'o-', color='tab:orange', label="Mean (FB)", zorder=3)
    ax1.axhline(ed_val.real, color='black', ls='--', label="ED")
    ax1.plot(0, extrap.real, 'D', color='tab:purple', markersize=8, label="ZNE (λ→0)")

    # Fit curve
    order = min(1, len(sf) - 1)
    coeffs = np.polyfit(sf, vals.real, order)
    x_fit = np.linspace(0, max(sf) * 1.1, 100)
    ax1.plot(x_fit, np.polyval(coeffs, x_fit), ':', color='tab:purple', alpha=0.5)

    ax1.set_xlabel("Noise scale factor λ")
    ax1.set_ylabel(f"Re[H_eff[{i},{j}]]")
    ax1.set_title(f"Real part")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Imaginary part
    ax2.plot(sf, vals.imag, 'o-', color='tab:orange', label="Mean (FB)", zorder=3)
    ax2.axhline(ed_val.imag, color='black', ls='--', label="ED")
    ax2.plot(0, extrap.imag, 'D', color='tab:purple', markersize=8, label="ZNE (λ→0)")

    coeffs_i = np.polyfit(sf, vals.imag, order)
    ax2.plot(x_fit, np.polyval(coeffs_i, x_fit), ':', color='tab:purple', alpha=0.5)

    ax2.set_xlabel("Noise scale factor λ")
    ax2.set_ylabel(f"Im[H_eff[{i},{j}]]")
    ax2.set_title(f"Imaginary part")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"ZNE Extrapolation: N={N}, H_eff[{i},{j}]", fontsize=13)
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(exist_ok=True, parents=True)
        fname = FIG_DIR / f"zne_extrap_N{N}_elem{i}{j}_j{int(J*100)}_hl{int(hl*100)}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


if __name__ == "__main__":
    zne_generate(J=1.0, hl=0.0, cluster_pairs=[(5, 4)], n_repetitions=10)
    zne_plot_single(N=5, J=1.0, hl=0.0)
    zne_plot_single(N=4, J=1.0, hl=0.0)
    zne_plot_nlce(cluster_pairs=[(5, 4)], J=1.0, hl=0.0)