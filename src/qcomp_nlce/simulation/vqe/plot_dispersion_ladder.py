"""
plot_dispersion_ladder.py
=========================
Load saved H_eff data from generate_dispersion_data_ladder.py and plot
ladder dispersion along the high-symmetry path in the Brillouin zone:

    (pi,0) -> (pi,pi) -> (0,0) -> (pi,0) -> (pi/2,pi/2)

NLCE is simply E(Nx2) - E((N-1)x2).

Data layout (under res_sim/dispersion_data_ladder/j{J}_hl{hl}/{n}x2/):
    H_eff_ED.npy, H_eff_SV.npy
    H_eff_sv_shots_rep{i}.npy
    gs_energy.npz, metadata.npz

Usage:
    python plot_dispersion_ladder.py
    python plot_dispersion_ladder.py --J 0.2 --cluster-pairs 3,2
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from qcomp_nlce.utils.dispersion import dispersion_ladder_thermodynamic

RES_SIM = Path(__file__).parent.parent.parent / "res_sim"
DATA_DIR = RES_SIM / "dispersion_data_ladder"
FIG_DIR = RES_SIM / "figures"


def _cluster_dir(n_rungs: int, J: float, hl: float) -> Path:
    return DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"{n_rungs}x2"


# ── High-symmetry path ──────────────────────────────────────────────────────

def high_symmetry_path(n_points_per_segment: int = 50):
    """Build the high-symmetry path:
        (pi,0) -> (pi,pi) -> (0,0) -> (pi,0) -> (pi/2,pi/2)

    Returns:
        kxs, kys: arrays of (kx, ky) along the path
        tick_positions: cumulative-distance positions of the corner points
        tick_labels: labels for the corner points
    """
    corners = [
        (np.pi, 0),
        (np.pi, np.pi),
        (0, 0),
        (np.pi, 0),
        (np.pi / 2, np.pi / 2),
    ]
    labels = [
        r"$(\pi,0)$",
        r"$(\pi,\pi)$",
        r"$(0,0)$",
        r"$(\pi,0)$",
        r"$(\pi/2,\pi/2)$",
    ]

    kxs, kys, dists = [], [], []
    cum_dist = 0.0
    tick_positions = [0.0]

    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        seg_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Don't duplicate the start point of each segment (except the very first)
        n = n_points_per_segment
        t = np.linspace(0, 1, n, endpoint=(i == len(corners) - 2))
        if i > 0:
            t = t[1:]  # skip duplicate corner

        for ti in t:
            kxs.append(x0 + ti * (x1 - x0))
            kys.append(y0 + ti * (y1 - y0))
            dists.append(cum_dist + ti * seg_len)

        cum_dist += seg_len
        tick_positions.append(cum_dist)

    return (
        np.array(kxs),
        np.array(kys),
        np.array(dists),
        tick_positions,
        labels,
    )


# ── Loading ──────────────────────────────────────────────────────────────────

def load_heff(n_rungs: int, J: float, hl: float) -> dict:
    cdir = _cluster_dir(n_rungs, J, hl)
    if not cdir.exists():
        print(f"Warning: Directory {cdir} does not exist. Skipping {n_rungs}x2.")
        return {"ed": None, "sv": None, "sv_shots": [], "fb": []}

    entry = {"ed": None, "sv": None, "sv_shots": [], "fb": []}

    ed_path = cdir / "H_eff_ED.npy"
    if ed_path.exists():
        entry["ed"] = np.load(ed_path)

    sv_path = cdir / "H_eff_SV.npy"
    if sv_path.exists():
        entry["sv"] = np.load(sv_path)

    for i in range(1000):
        p = cdir / f"H_eff_sv_shots_rep{i}.npy"
        if not p.exists():
            break
        entry["sv_shots"].append(np.load(p))

    for i in range(1000):
        p = cdir / f"H_eff_fake_backend_rep{i}.npy"
        if not p.exists():
            break
        entry["fb"].append(np.load(p))

    label = f"{n_rungs}x2"
    print(f"{label}: ED={'yes' if entry['ed'] is not None else 'no'}, "
          f"SV={'yes' if entry['sv'] is not None else 'no'}, "
          f"SV_shots={len(entry['sv_shots'])} reps, "
          f"FB={len(entry['fb'])} reps")

    return entry


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_nlce_ladder(
    cluster_pairs: list[tuple[int, int]],
    J: float,
    hl: float,
    save: bool = True,
    n_points_per_segment: int = 50,
):
    """Plot NLCE ladder dispersion along the high-symmetry BZ path."""
    kxs, kys, dists, tick_pos, tick_labels = high_symmetry_path(n_points_per_segment)

    # Load all needed cluster sizes
    all_n = sorted(set(n for pair in cluster_pairs for n in pair))
    cache = {n: load_heff(n, J, hl) for n in all_n}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color gradient for different N_max (like the reference plot)
    ed_cmap = plt.cm.Blues
    sv_cmap = plt.cm.Reds
    n_pairs = len(cluster_pairs)

    for idx, (n_big, n_small) in enumerate(cluster_pairs):
        big, small = cache[n_big], cache[n_small]
        n_max = 2 * n_big  # total qubits in biggest cluster
        frac = (idx + 1) / (n_pairs + 1)  # color fraction (darker = larger)

        # ED
        if big["ed"] is not None and small["ed"] is not None:
            disp = [dispersion_ladder_thermodynamic(kx, ky, big["ed"], small["ed"])
                    for kx, ky in zip(kxs, kys)]
            ax.plot(dists, disp, '-', color=ed_cmap(0.3 + 0.6 * frac),
                    linewidth=1.5, label=f"NLCE+ED N_max={n_max}")

        # Exact SV (VQE)
        if big["sv"] is not None and small["sv"] is not None:
            disp = [dispersion_ladder_thermodynamic(kx, ky, big["sv"], small["sv"])
                    for kx, ky in zip(kxs, kys)]
            ax.plot(dists, disp, ':', color=sv_cmap(0.3 + 0.6 * frac),
                    linewidth=1.5, label=f"NLCE+VQE N_max={n_max}")

        # SV shots (mean + errorbar)
        if big["sv_shots"] and small["sv_shots"]:
            n_reps = min(len(big["sv_shots"]), len(small["sv_shots"]))
            all_disp = np.array([
                [dispersion_ladder_thermodynamic(kx, ky, big["sv_shots"][r], small["sv_shots"][r])
                 for kx, ky in zip(kxs, kys)]
                for r in range(n_reps)
            ])
            mean = np.mean(all_disp, axis=0)
            sem = (np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps)
                   if n_reps > 1 else np.zeros_like(mean))
            ax.fill_between(dists, mean - sem, mean + sem,
                            color=sv_cmap(0.3 + 0.6 * frac), alpha=0.15)

    # Vertical lines at high-symmetry points
    for tp in tick_pos:
        ax.axvline(tp, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(tick_pos[0], tick_pos[-1])
    ax.set_xlabel(r"$(k_x, k_y)$")
    ax.set_ylabel(r"$\omega(k_x, k_y)$")
    ax.set_title(f"1QP Dispersion — TFIM Ladder (J={J})")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(exist_ok=True, parents=True)
        fname = FIG_DIR / f"dispersion_ladder_nlce_j{int(J*100)}_hl{int(hl*100)}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return (int(a), int(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saved ladder dispersion data")
    parser.add_argument("--J", type=float, default=0.328)
    parser.add_argument("--hl", type=float, default=0.0)
    parser.add_argument("--cluster-pairs", nargs="+", type=str, default=None,
                        help="Cluster pairs as N_big,N_small (e.g. 3,2 4,3)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    pairs = [_parse_pair(p) for p in args.cluster_pairs] if args.cluster_pairs else [(3, 2)]

    plot_nlce_ladder(pairs, args.J, args.hl, save=not args.no_save)
