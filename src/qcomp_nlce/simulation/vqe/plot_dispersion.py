

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from qcomp_nlce.utils.dispersion import cluster_dispersion, dispersion_1d_thermodynamic
from qcomp_nlce.utils.effective_hamiltonian import H_effective
from qcomp_nlce.utils.stats import complex_normal

RES_SIM = Path(__file__).parent.parent.parent / "res_sim"
DATA_DIR = RES_SIM / "dispersion_data"
FIG_DIR = RES_SIM / "figures"


def _cluster_dir(N: int, J: float, hl: float) -> Path:
    return DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"N{N}"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_heff(N: int, J: float, hl: float) -> dict:
    """
    Load all H_eff matrices for cluster size N.

    Returns dict with keys: "ed", "sv", "sv_shots" (list), "fb" (list).
    Missing files are None / empty list.
    """
    cdir = _cluster_dir(N, J, hl)
    if not cdir.exists():
        print(f"Warning: Directory {cdir} does not exist. Skipping N={N}.")
        return {"ed": None, "sv": None, "sv_shots": [], "fb": []}
    
    entry = {"ed": None, "sv": None, "sv_shots": [], "fb": []}

    ed_path = cdir / "H_eff_ED.npy"
    if ed_path.exists():
        entry["ed"] = np.load(ed_path)

    sv_path = cdir / "H_eff_SV.npy"
    if sv_path.exists():
        entry["sv"] = np.load(sv_path)

    # Load all reps
    for i in range(1000):  # generous upper bound
        p = cdir / f"H_eff_sv_shots_rep{i}.npy"
        if not p.exists():
            break
        entry["sv_shots"].append(np.load(p))

    for i in range(1000):
        p = cdir / f"H_eff_fake_backend_rep{i}.npy"
        if not p.exists():
            break
        entry["fb"].append(np.load(p))

    print(f"N={N}: ED={'yes' if entry['ed'] is not None else 'no'}, "
          f"SV={'yes' if entry['sv'] is not None else 'no'}, "
          f"SV_shots={len(entry['sv_shots'])} reps, "
          f"FB={len(entry['fb'])} reps")

    return entry


def load_metadata(N: int, J: float, hl: float) -> dict:
    cdir = _cluster_dir(N, J, hl)
    meta_path = cdir / "metadata.npz"
    if meta_path.exists():
        return dict(np.load(meta_path, allow_pickle=True))
    return {}


def load_gs_energies(N: int, J: float, hl: float) -> dict:
    cdir = _cluster_dir(N, J, hl)
    gs_path = cdir / "gs_energy.npz"
    if gs_path.exists():
        d = np.load(gs_path)
        return {"E_ED": float(d["E_ED"]), "E_SV": float(d["E_SV"])}
    return {}


def load_mc_ingredients(N: int, J: float, hl: float, backend: str = "fake_backend") -> list[dict]:
    """
    Load MC error propagation ingredients for cluster N.
    Returns list of dicts (one per rep), each with keys:
    expect, expect_std, overlap, overlap_std, gs_energy, gs_std.
    """
    cdir = _cluster_dir(N, J, hl)
    prefix = "mc_fake_backend" if backend == "fake_backend" else "mc_sv_shots"
    ingredients = []
    for i in range(1000):
        p = cdir / f"{prefix}_rep{i}.npz"
        if not p.exists():
            break
        d = np.load(p)
        n_shots = int(d["n_shots"]) if "n_shots" in d else 200
        ingredients.append({
            "expect": d["expect"],
            "expect_std": d["expect_std"],
            "overlap": d["overlap"],
            "overlap_std": d["overlap_std"],
            "gs_energy": float(d["gs_energy"]),
            "gs_std": float(d["gs_std"]),
            "n_shots": n_shots,
        })
    return ingredients


# ── Monte Carlo error propagation ────────────────────────────────────────────

def _mc_sample_heff(mc: dict) -> np.ndarray:
    """Draw one MC sample of H_eff from measurement uncertainties.

    Applies to_eotm correction: expect_std and overlap_std are raw stds,
    so we divide by sqrt(n_shots) to get error-on-the-mean.
    gs_std is already EOTM from subspace sampling.
    """
    eotm_fac = 1.0 / np.sqrt(mc["n_shots"])
    expect = complex_normal(mc["expect"], mc["expect_std"] * eotm_fac)
    overlap = complex_normal(mc["overlap"], mc["overlap_std"] * eotm_fac)
    gs = complex_normal(
        np.array(mc["gs_energy"] + 0j),
        np.array(mc["gs_std"] + 0j),
    )
    return H_effective(float(np.real(gs)), expect, overlap)


def mc_dispersion_nlce(
    ks: np.ndarray,
    mc_big: dict, mc_small: dict,
    n_samples: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MC error propagation for NLCE dispersion (big - small).
    Uses a single rep's MC ingredients for each cluster.
    Returns (mean, std) arrays over k-points.
    """
    sampled = np.zeros((n_samples, len(ks)))
    for n in range(n_samples):
        heff_b = _mc_sample_heff(mc_big)
        heff_s = _mc_sample_heff(mc_small)
        sampled[n] = [dispersion_1d_thermodynamic(k, heff_b, heff_s) for k in ks]
    return np.mean(sampled, axis=0), np.std(sampled, axis=0)


def mc_dispersion_single(
    N: int, ks: np.ndarray,
    mc: dict,
    n_samples: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MC error propagation for single-cluster dispersion.
    Returns (mean, std) arrays over k-points.
    """
    sampled = np.zeros((n_samples, len(ks)))
    for n in range(n_samples):
        heff = _mc_sample_heff(mc)
        sampled[n] = [np.real(cluster_dispersion(N, heff, k)) for k in ks]
    return np.mean(sampled, axis=0), np.std(sampled, axis=0)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_nlce(
    cluster_pairs: list[tuple[int, int]],
    J: float, hl: float,
    save: bool = True,
    n_k_points: int = 20,
    n_mc_samples: int = 10000,
):
    """Plot NLCE dispersion (big - small subtraction) with errorbar style."""
    ks = np.linspace(0, np.pi, n_k_points)
    ks_dense = np.linspace(0, np.pi, 200)

    # Load all needed cluster sizes
    all_n = sorted(set(n for pair in cluster_pairs for n in pair))
    cache = {N: load_heff(N, J, hl) for N in all_n}

    meta = load_metadata(all_n[0], J, hl)
    n_shots = meta.get("n_shots", "?")
    n_reps_meta = meta.get("n_repetitions", "?")

    fig, ax = plt.subplots(figsize=(8, 6))

    COLOR_ED = 'black'
    COLOR_SV = 'tab:blue'
    COLOR_SV_SHOTS = 'tab:green'
    COLOR_FB = 'tab:orange'
    COLOR_MC = 'tab:red'

    if np.isclose(hl, 0.0):
        from qcomp_nlce.utils.dispersion import exact_dispersion_1qp_1d
        ax.plot(ks_dense, exact_dispersion_1qp_1d(J, ks_dense), '-', color='gray',
                linewidth=2.5, label="Exact analytic (hl=0)", zorder=0)

    for idx, (n_big, n_small) in enumerate(cluster_pairs):
        big, small = cache[n_big], cache[n_small]

        # ED — line + small square markers
        if big["ed"] is not None and small["ed"] is not None:
            disp = [dispersion_1d_thermodynamic(k, big["ed"], small["ed"]) for k in ks]
            ax.plot(ks, disp, '--s', color=COLOR_ED, linewidth=1.5, markersize=4,
                    label="NLCE ED", zorder=5)

        # Exact SV — line + small circle markers
        if big["sv"] is not None and small["sv"] is not None:
            disp = [dispersion_1d_thermodynamic(k, big["sv"], small["sv"]) for k in ks]
            ax.plot(ks, disp, '-o', color=COLOR_SV, linewidth=1.5, markersize=4,
                    label="Statevector", zorder=4)

        # SV shots — line + errorbar with small down-triangle markers
        if big["sv_shots"] and small["sv_shots"]:
            n_reps = min(len(big["sv_shots"]), len(small["sv_shots"]))
            all_disp = np.array([
                [dispersion_1d_thermodynamic(k, big["sv_shots"][r], small["sv_shots"][r]) for k in ks]
                for r in range(n_reps)
            ])
            mean = np.mean(all_disp, axis=0)
            sem = np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else np.zeros_like(mean)
            ax.errorbar(ks, mean, yerr=sem, fmt='-v', color=COLOR_SV_SHOTS, markersize=4,
                        capsize=3, linewidth=1.5, label="SV shots", zorder=3)

        # Fake backend — line + errorbar with small up-triangle markers
        if big["fb"] and small["fb"]:
            n_reps = min(len(big["fb"]), len(small["fb"]))
            all_disp = np.array([
                [dispersion_1d_thermodynamic(k, big["fb"][r], small["fb"][r]) for k in ks]
                for r in range(n_reps)
            ])
            mean = np.mean(all_disp, axis=0)
            sem = np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else np.zeros_like(mean)
            ax.errorbar(ks, mean, yerr=sem, fmt='-^', color=COLOR_FB, markersize=4,
                        capsize=3, linewidth=1.5, label="Fake Backend", zorder=2)

        # MC error propagation (fake backend, rep 0) — separate curve
        mc_big = load_mc_ingredients(n_big, J, hl, "fake_backend")
        mc_small = load_mc_ingredients(n_small, J, hl, "fake_backend")
        if mc_big and mc_small:
            print(f"  Computing MC error bars ({n_big},{n_small})...")
            mc_mean, mc_std = mc_dispersion_nlce(ks, mc_big[0], mc_small[0], n_samples=n_mc_samples)
            ax.errorbar(ks, mc_mean, yerr=mc_std, fmt='--d', color=COLOR_MC, markersize=4,
                        capsize=3, linewidth=1.5, label="FB (MC errorbars)", zorder=1)

    ax.set_xlabel("k")
    ax.set_ylabel("ω(k)")
    ax.set_title(f"NLCE 1qp Dispersion ({n_shots} shots, {n_reps_meta} reps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(exist_ok=True, parents=True)
        fname = FIG_DIR / f"dispersion_nlce_j{int(J*100)}_hl{int(hl*100)}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


def plot_single_cluster(N: int, J: float, hl: float, save: bool = True, n_k_points: int = 11, n_mc_samples: int = 10000):
    """
    Plot dispersion for a single cluster of size N (no NLCE subtraction).
    Uses marker+errorbar style matching collaborator plots.
    """
    ks = np.linspace(0, np.pi, n_k_points)
    entry = load_heff(N, J, hl)

    fig, ax = plt.subplots(figsize=(8, 6))

    COLOR_ED = 'black'
    COLOR_SV = 'tab:blue'
    COLOR_SV_SHOTS = 'tab:green'
    COLOR_FB = 'tab:orange'
    COLOR_MC = 'tab:red'

    # ED — line + small square markers
    if entry["ed"] is not None:
        heff_ed = entry["ed"]
    else:
        from qcomp_nlce.utils.effective_hamiltonian import H_effective_ed
        from qcomp_nlce.utils.hamiltonian import TFIM
        ham = TFIM(N, h=1, J=J, periodic=False, longitudinal_field=hl)
        heff_ed = H_effective_ed(ham)
    disp_ed = [np.real(cluster_dispersion(N, heff_ed, k)) for k in ks]
    ax.plot(ks, disp_ed, '--s', color=COLOR_ED, linewidth=1.5, markersize=4,
            label="ED", zorder=5)

    # Exact SV — line + small circle markers
    if entry["sv"] is not None:
        disp_sv = [np.real(cluster_dispersion(N, entry["sv"], k)) for k in ks]
        ax.plot(ks, disp_sv, '-o', color=COLOR_SV, linewidth=1.5, markersize=4,
                label="Statevector", zorder=4)

    # SV shots — line + errorbar with small down-triangle markers
    if entry["sv_shots"]:
        all_disp = np.array([
            [np.real(cluster_dispersion(N, h, k)) for k in ks]
            for h in entry["sv_shots"]
        ])
        mean = np.mean(all_disp, axis=0)
        n_reps = len(all_disp)
        sem = np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else np.zeros_like(mean)
        ax.errorbar(ks, mean, yerr=sem, fmt='-v', color=COLOR_SV_SHOTS, markersize=4,
                    capsize=3, linewidth=1.5, label="SV shots", zorder=3)

    # Fake backend — line + errorbar with small up-triangle markers
    if entry["fb"]:
        all_disp = np.array([
            [np.real(cluster_dispersion(N, h, k)) for k in ks]
            for h in entry["fb"]
        ])
        mean = np.mean(all_disp, axis=0)
        n_reps = len(all_disp)
        sem = np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else np.zeros_like(mean)
        ax.errorbar(ks, mean, yerr=sem, fmt='-^', color=COLOR_FB, markersize=4,
                    capsize=3, linewidth=1.5, label="Fake Backend", zorder=2)

    # MC error propagation (fake backend, rep 0) — separate curve
    mc_list = load_mc_ingredients(N, J, hl, "fake_backend")
    if mc_list:
        print(f"  Computing MC error bars for N={N}...")
        mc_mean, mc_std = mc_dispersion_single(N, ks, mc_list[0], n_samples=n_mc_samples)
        ax.errorbar(ks, mc_mean, yerr=mc_std, fmt='--d', color=COLOR_MC, markersize=4,
                    capsize=3, linewidth=1.5, label="FB (MC errorbars)", zorder=1)

    meta = load_metadata(N, J, hl)
    n_shots = meta.get("n_shots", "?")
    n_reps_meta = meta.get("n_repetitions", "?")

    ax.set_xlabel("k")
    ax.set_ylabel("ω(k)")
    ax.set_title(f"1qp Dispersion: N={N} ({n_shots} shots, {n_reps_meta} reps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(exist_ok=True, parents=True)
        fname = FIG_DIR / f"dispersion_single_N{N}_j{int(J*100)}_hl{int(hl*100)}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


def print_comparison_table(
    cluster_pairs: list[tuple[int, int]],
    J: float, hl: float,
):
    """Print dispersion comparison at selected k-points."""
    ks = np.linspace(0, np.pi, 100)
    k_indices = np.linspace(0, len(ks) - 1, 11, dtype=int)

    all_n = sorted(set(n for pair in cluster_pairs for n in pair))
    cache = {N: load_heff(N, J, hl) for N in all_n}

    for (n_big, n_small) in cluster_pairs:
        big, small = cache[n_big], cache[n_small]

        has_ed = big["ed"] is not None and small["ed"] is not None
        has_sv = big["sv"] is not None and small["sv"] is not None
        n_svs = min(len(big["sv_shots"]), len(small["sv_shots"])) if big["sv_shots"] and small["sv_shots"] else 0
        n_fb = min(len(big["fb"]), len(small["fb"])) if big["fb"] and small["fb"] else 0

        print(f"\n{'='*90}")
        print(f"Dispersion Comparison ({n_big},{n_small}) -- J={J}, hl={hl}")
        print(f"{'='*90}")
        print(f"{'k/pi':<8} {'ED':<12} {'SV exact':<12} {'SV shots':<12} {'SV SEM':<12} {'FB mean':<12} {'FB SEM':<12} {'FB-ED':<12}")
        print(f"{'-'*90}")

        for ki in k_indices:
            k = ks[ki]
            k_pi = k / np.pi

            ed_val = dispersion_1d_thermodynamic(k, big["ed"], small["ed"]) if has_ed else float('nan')
            sv_val = dispersion_1d_thermodynamic(k, big["sv"], small["sv"]) if has_sv else float('nan')

            if n_svs > 0:
                vals = [dispersion_1d_thermodynamic(k, big["sv_shots"][r], small["sv_shots"][r]) for r in range(n_svs)]
                svs_mean, svs_sem = np.mean(vals), (np.std(vals, ddof=1) / np.sqrt(n_svs) if n_svs > 1 else 0.0)
            else:
                svs_mean = svs_sem = float('nan')

            if n_fb > 0:
                vals = [dispersion_1d_thermodynamic(k, big["fb"][r], small["fb"][r]) for r in range(n_fb)]
                fb_mean, fb_sem = np.mean(vals), (np.std(vals, ddof=1) / np.sqrt(n_fb) if n_fb > 1 else 0.0)
                fb_diff = fb_mean - ed_val if has_ed else float('nan')
            else:
                fb_mean = fb_sem = fb_diff = float('nan')

            print(f"{k_pi:<8.2f} {ed_val:<12.6f} {sv_val:<12.6f} {svs_mean:<12.6f} {svs_sem:<12.6f} {fb_mean:<12.6f} {fb_sem:<12.6f} {fb_diff:<+12.6f}")

        print(f"{'='*90}")


def _parse_pair(s: str) -> tuple[int, int]:
    """Parse '4,3' into (4, 3)."""
    a, b = s.split(",")
    return (int(a), int(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saved dispersion data")
    parser.add_argument("--J", type=float, default=0.5)
    parser.add_argument("--hl", type=float, default=0.1)
    parser.add_argument("--cluster-pairs", nargs="+", type=str, default=None,
                        help="Cluster pairs as N_big,N_small (e.g. 4,3 5,4)")
    parser.add_argument("--single-cluster", type=int, default=None,
                        help="Plot single cluster N (no NLCE subtraction)")
    parser.add_argument("--table", action="store_true", help="Print comparison table")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    pairs = [_parse_pair(p) for p in args.cluster_pairs] if args.cluster_pairs else [(4, 3)]

    if args.single_cluster is not None:
        plot_single_cluster(args.single_cluster, args.J, args.hl, save=not args.no_save)
    elif args.table:
        print_comparison_table(pairs, args.J, args.hl)
    else:
        plot_nlce(pairs, args.J, args.hl, save=not args.no_save)


