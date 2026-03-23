

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from src.qcomp_nlce.utils.dispersion import dispersion_ladder_thermodynamic

RES_SIM = Path(__file__).parent.parent.parent / "res_sim"
DATA_DIR = RES_SIM / "dispersion_data_ladder"
FIG_DIR = RES_SIM / "figures"


def _cluster_dir(n_rungs: int, J: float, hl: float) -> Path:
    return DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"{n_rungs}x2"


def k_vals(n_points: int = 100):
   
    ks = np.linspace(-np.pi, np.pi, n_points)
    tick_positions = [-np.pi, 0.0, np.pi]
    tick_labels = [r"$-\pi$", r"$0$", r"$\pi$"]
    return ks, tick_positions, tick_labels


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
    n_points: int = 100,
):
    """Plot NLCE ladder dispersion along k = 0 to pi."""
    ks, tick_pos, tick_labels = k_vals(n_points)

    # Load all needed cluster sizes
    all_n = sorted(set(n for pair in cluster_pairs for n in pair))
    cache = {n: load_heff(n, J, hl) for n in all_n}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Style constants matching reference figure
    COLORS = {
        "ed": "black",       
        "sv": "blue",       
        "sv_shots":"orange", 
        "fb": "green",       
    }
    MARKERS = {
        "ed": "s",             
        "sv": "o",             
        "sv_shots": "^",       
        "fb": "v",             
    }
    LW = 2.0
    MS = 6
    n_k = len(ks)
    marker_every = max(1, n_k // 10)

    for idx, (n_big, n_small) in enumerate(cluster_pairs):
        big, small = cache[n_big], cache[n_small]

        # ED — two bands
        if big["ed"] is not None and small["ed"] is not None:
            disp = np.array([dispersion_ladder_thermodynamic(k, big["ed"], small["ed"])
                             for k in ks])  # shape (n_k, 2)
            for band in range(disp.shape[1]):
                label = "NLCE+ED" if band == 0 else None
                ax.plot(ks, disp[:, band], '-', color=COLORS["ed"],
                        marker=MARKERS["ed"], markevery=marker_every, markersize=MS,
                        linewidth=LW, label=label)

        # Exact SV (VQE) — two bands
        if big["sv"] is not None and small["sv"] is not None:
            disp = np.array([dispersion_ladder_thermodynamic(k, big["sv"], small["sv"])
                             for k in ks])
            for band in range(disp.shape[1]):
                label = "Statevector" if band == 0 else None
                ax.plot(ks, disp[:, band], '-', color=COLORS["sv"],
                        marker=MARKERS["sv"], markevery=marker_every, markersize=MS,
                        linewidth=LW, label=label)

        # SV shots (mean + error bars) — two bands
        if big["sv_shots"] and small["sv_shots"]:
            n_reps = min(len(big["sv_shots"]), len(small["sv_shots"]))
            all_disp = np.array([
                [dispersion_ladder_thermodynamic(k, big["sv_shots"][r], small["sv_shots"][r])
                 for k in ks]
                for r in range(n_reps)
            ])  # shape (n_reps, n_k, 2)
            mean = np.mean(all_disp, axis=0)  # (n_k, 2)
            sem = (np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps)
                   if n_reps > 1 else np.zeros_like(mean))
            marker_idx = np.arange(0, n_k, marker_every)
            for band in range(mean.shape[1]):
                label = "Statevector (shots)" if band == 0 else None
                ax.plot(ks, mean[:, band], '-', color=COLORS["sv_shots"],
                        linewidth=LW, alpha=0.7)
                ax.errorbar(ks[marker_idx], mean[marker_idx, band],
                            yerr=sem[marker_idx, band],
                            fmt=MARKERS["sv_shots"], color=COLORS["sv_shots"],
                            markersize=MS, capsize=3, capthick=1.5,
                            linewidth=0, elinewidth=1.5, label=label)

        # Fake backend (mean + error bars) — two bands
        if big["fb"] and small["fb"]:
            n_reps = min(len(big["fb"]), len(small["fb"]))
            all_disp = np.array([
                [dispersion_ladder_thermodynamic(k, big["fb"][r], small["fb"][r])
                 for k in ks]
                for r in range(n_reps)
            ])  # shape (n_reps, n_k, 2)
            mean = np.mean(all_disp, axis=0)
            sem = (np.std(all_disp, axis=0, ddof=1) / np.sqrt(n_reps)
                   if n_reps > 1 else np.zeros_like(mean))
            marker_idx = np.arange(0, n_k, marker_every)
            for band in range(mean.shape[1]):
                label = "Fake Backend" if band == 0 else None
                ax.plot(ks, mean[:, band], '-', color=COLORS["fb"],
                        linewidth=LW, alpha=0.7)
                ax.errorbar(ks[marker_idx], mean[marker_idx, band],
                            yerr=sem[marker_idx, band],
                            fmt=MARKERS["fb"], color=COLORS["fb"],
                            markersize=MS, capsize=3, capthick=1.5,
                            linewidth=0, elinewidth=1.5, label=label)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_xlim(tick_pos[0], tick_pos[-1])
    ax.set_xlabel(r"$k$", fontsize=13)
    ax.set_ylabel(r"$\omega(k)$", fontsize=13)
    ax.set_title(f"1QP Dispersion — TFIM Ladder (J={J})", fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(exist_ok=True, parents=True)
        fname = FIG_DIR / f"dispersion_ladder_nlce_j{int(J*100)}.png"
        fig.savefig(fname, dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return (int(a), int(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saved ladder dispersion data")
    parser.add_argument("--J", type=float, default=0.2)
    parser.add_argument("--hl", type=float, default=0.0)
    parser.add_argument("--cluster-pairs", nargs="+", type=str, default=None,
                        help="Cluster pairs as N_big,N_small (e.g. 3,2 4,3)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    pairs = [_parse_pair(p) for p in args.cluster_pairs] if args.cluster_pairs else [(3, 2)]

    plot_nlce_ladder(pairs, args.J, args.hl, save=not args.no_save)
